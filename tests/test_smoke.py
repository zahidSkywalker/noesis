"""Noesis QA smoke suite — proves every pipe before GPU money is spent.

Run:  python -m pytest tests/ -q   (or: python tests/test_smoke.py)
Covers:
  1. tokenizer round-trip (byte-level = lossless)
  2. tokenizer trained on real-ish text + encode/decode sanity
  3. model forward/backward (shapes, finite loss, grads exist)
  4. KV-cache consistency: cached generation == full-recompute greedy
  5. checkpoint save -> resume -> training continues without NaN
  6. quantizer: int8 payload ~4x smaller
  7. val split: train stream never samples the held-out tail, [val] evals
     fire in a real train.main() run, and PRE-val checkpoints resume clean
  8. val split hardening: an eval that RAISES must not kill training —
     session completes, model returns to train mode, best_val stays None
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noesis.tokenizer import BPETokenizer
from noesis.model import ModelConfig, NoesisModel
from noesis.data import binarize, TokenStream
from noesis.generate import generate
from noesis.train import save_checkpoint, load_checkpoint, cosine_lr

SAMPLE = ("Alice was beginning to get very tired of sitting by her sister on the bank, "
          "and of having nothing to do. So she was considering in her own mind whether "
          "the pleasure of making a daisy-chain would be worth the trouble of getting up "
          "and picking the daisies. There was nothing so VERY remarkable in that.\n" * 40)


def test_tokenizer_roundtrip_lossless():
    tok = BPETokenizer()
    text = "Hello, 世界! emoji 🙂 and symbols ×÷~`\n\ttabs and  spaces   kept."
    ids = tok.encode(text)
    assert tok.decode(ids) == text, "byte-level BPE must round-trip losslessly"


def test_tokenizer_train_and_roundtrip():
    tok = BPETokenizer.train(SAMPLE, vocab_size=600, verbose=False)
    assert 256 < tok.vocab_size <= 600
    assert tok.decode(tok.encode(SAMPLE)) == SAMPLE
    # common words should have become single tokens
    ids = tok.encode(" the ")
    assert len(ids) <= 3, "frequent word 'the' should merge into few tokens"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "vocab.json")
        tok.save(p)
        tok2 = BPETokenizer.load(p)
        assert tok2.vocab_size == tok.vocab_size
        assert tok2.encode(SAMPLE) == tok.encode(SAMPLE)


def test_model_forward_backward():
    cfg = ModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16,
                      vocab_size=512, max_seq_len=64, mlp_hidden=128)
    model = NoesisModel(cfg)
    x = torch.randint(0, 512, (2, 32))
    y = torch.randint(0, 512, (2, 32))
    logits, loss = model(x, targets=y)
    assert logits.shape == (2, 32, 512)
    assert torch.isfinite(loss), "loss must be finite"
    loss.backward()
    grads = [p.grad is not None for p in model.parameters()]
    assert all(grads), "every parameter must receive grad"
    n = sum(p.numel() for p in model.parameters())
    assert 100_000 < n < 2_000_000, f"nano model should be ~1M params, got {n}"


def test_production_configs_forward_backward():
    """Regression for the M4 launch bug: base_125m shipped n_kv_heads=8 under
    n_heads=12 — 12//8==1 silently skipped the GQA repeat and the very first
    GPU forward died with a matmul shape error. Every shipped config must
    survive a real forward/backward pass before it goes near a GPU."""
    import glob as _glob
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = sorted(_glob.glob(os.path.join(root, "configs", "base*.json")))
    assert paths, "no production configs found"
    for path in paths:
        with open(path) as f:
            full = json.load(f)
        cfg = ModelConfig.from_dict(full["model"])
        assert cfg.n_heads % cfg.n_kv_heads == 0, \
            f"{path}: invalid GQA ratio {cfg.n_heads}/{cfg.n_kv_heads}"
        model = NoesisModel(cfg)
        x = torch.randint(0, cfg.vocab_size, (1, 32))
        logits, loss = model(x, targets=x)
        assert logits.shape == (1, 32, cfg.vocab_size)
        assert torch.isfinite(loss), f"{path}: non-finite loss"
        loss.backward()
        n = sum(p.numel() for p in model.parameters())
        assert 100_000_000 < n < 200_000_000, f"{path}: expected ~125M params, got {n}"


def test_kv_cache_matches_full_recompute():
    torch.manual_seed(0)
    cfg = ModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16,
                      vocab_size=512, max_seq_len=64, mlp_hidden=128)
    model = NoesisModel(cfg).eval()

    prompt = torch.randint(0, 512, (1, 8))
    # greedy full-recompute baseline
    seq = prompt.clone()
    for _ in range(12):
        logits, _ = model(seq)
        seq = torch.cat([seq, torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)], dim=1)
    # cached generation
    caches = model.init_caches(1, "cpu")
    logits, _ = model(prompt, caches=caches)
    nid = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    seq2 = torch.cat([prompt, nid], dim=1)
    for _ in range(11):
        logits, _ = model(nid, caches=caches)
        nid = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        seq2 = torch.cat([seq2, nid], dim=1)
    assert torch.equal(seq, seq2), "KV-cache path must match full recompute exactly"


def test_sdpa_matches_manual_attention():
    """v7 OOM fix: training now uses fused SDPA; decode keeps manual
    global-causal attention. Both paths must agree on identical tokens."""
    torch.manual_seed(1)
    cfg = ModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16,
                      vocab_size=512, max_seq_len=64, mlp_hidden=128)
    model = NoesisModel(cfg).eval()
    x = torch.randint(0, 512, (1, 12))
    with torch.no_grad():
        full, _ = model(x)                       # SDPA path (no cache)
        caches = model.init_caches(1, "cpu")
        model(x[:, :5], caches=caches)           # warm cache on prefix
        cont, _ = model(x[:, 5:], caches=caches)  # manual path continuation
    assert torch.allclose(full[:, 5:], cont, atol=1e-4), \
        "SDPA and manual attention paths must agree on the same tokens"


def test_checkpoint_resume():
    cfg = ModelConfig(dim=48, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=12,
                      vocab_size=400, max_seq_len=32, mlp_hidden=96)
    model = NoesisModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 400, (2, 32))
    y = torch.randint(0, 400, (2, 32))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "latest.pt")
        save_checkpoint(p, model, opt, 10, cfg.to_dict(), "vocab.json", "data.bin", 10)
        _, cfg2, model2 = load_checkpoint(p, "cpu")
        assert cfg2.to_dict() == cfg.to_dict()
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        opt2.load_state_dict(opt.state_dict())
        _, loss = model2(x, targets=y)
        assert torch.isfinite(loss)


def test_resume_optimizer_binding():
    """v9 resume crash regression: train.py built the optimizer BEFORE resume,
    then load_checkpoint() rebinds `model` to a fresh object — backward filled
    the new object's grads while the optimizer/scaler only saw the original
    params (all grad=None) -> 'No inf checks were recorded for this optimizer'
    on the very first scaler.step() after resume. The real sequence must load
    weights into the LIVE model so grads land on optimizer-held tensors."""
    torch.manual_seed(0)
    cfg = ModelConfig(dim=32, n_layers=1, n_heads=2, n_kv_heads=1, head_dim=16,
                      vocab_size=64, max_seq_len=16, mlp_hidden=64)
    model = NoesisModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)  # optimizer BEFORE resume
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "latest.pt")
        save_checkpoint(p, model, opt, 3, cfg.to_dict(), "vocab.json", "data.bin", 3)
        # the exact resume sequence now used by train.main():
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
    # identity: every live param must be a tensor the optimizer actually holds
    opt_ids = {id(q) for g in opt.param_groups for q in g["params"]}
    assert all(id(q) in opt_ids for q in model.parameters()), \
        "optimizer must reference the live model's parameters after resume"
    # grads must flow into optimizer-held tensors
    _, loss = model(x, targets=y)
    loss.backward()
    held = [q for g in opt.param_groups for q in g["params"]]
    assert all(q.grad is not None for q in held), \
        "backward must populate grads on optimizer-held params (No-inf-checks regression)"


def test_train_resume_does_not_rebind_model():
    """Source-level pin: train.main() must never call load_checkpoint() (it
    returns a NEW model object -> orphans the optimizer) and must load the
    checkpoint into the live model via model.load_state_dict."""
    import inspect
    from noesis import train
    src = inspect.getsource(train.main)
    assert "load_checkpoint(" not in src, \
        "train.main must not use load_checkpoint() — it rebinds model and orphans the optimizer"
    assert "model.load_state_dict(ckpt[\"model\"])" in src, \
        "resume path must load weights into the live model object"


def test_val_region_isolation():
    """Val split guarantee #1: the train stream must NEVER sample the held-out
    tail, and the val stream must see ONLY the tail. Proven with a sentinel
    token id that exists solely in the tail."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tokens.bin")
        n_train, n_val = 50_000, 2_000
        arr = np.concatenate([
            np.random.default_rng(0).integers(0, 1000, size=n_train),
            np.full(n_val, 4095, dtype=np.int64),  # sentinel: tail only
        ]).astype(np.uint16)
        arr.tofile(p)
        SEQ, BS = 64, 8
        train = TokenStream(p, SEQ, BS, seed=1, end=n_train)
        val = TokenStream(p, SEQ, BS, seed=2, start=n_train)
        assert train.end == n_train and val.start == n_train
        for _ in range(200):  # train stream: sentinel must never appear
            x, y = train.next_batch()
            assert (x < 4095).all() and (y < 4095).all(), "train stream leaked into val tail"
        for _ in range(50):   # val stream: every position at/after the border
            x, y = val.next_batch()
            assert (x >= 4095).all() and (y >= 4095).all(), "val stream saw pre-border tokens"
        # default args reproduce the original whole-file behavior exactly
        full = TokenStream(p, SEQ, BS, seed=1)
        assert full.start == 0 and full.end == n_train + n_val
        # a region too small for even one window raises loudly, never mis-samples
        try:
            TokenStream(p, SEQ, BS, seed=3, start=n_train + n_val - 10)
            raise AssertionError("tiny val region must raise ValueError")
        except ValueError:
            pass


def test_val_eval_end_to_end_with_resume():
    """Val split guarantee #2: a real train.main() run prints [val] baseline +
    periodic lines, best_val rides in checkpoints, and a PRE-val checkpoint
    (best_val=None, i.e. what v17 saves) resumes cleanly — the exact
    v17 -> v18 upgrade path session 7 will take on the live run."""
    import subprocess
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory() as d:
        text_path = os.path.join(d, "corpus.txt")
        with open(text_path, "w") as f:
            f.write(SAMPLE)
        tok = BPETokenizer.train(SAMPLE, vocab_size=400)
        tok.save(os.path.join(d, "vocab.json"))
        n = binarize(tok, [text_path], os.path.join(d, "tokens.bin"))
        assert n >= 512, f"tiny corpus too small: {n} tokens"

        cfg = {
            "model": {"dim": 48, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
                      "head_dim": 12, "vocab_size": tok.vocab_size,
                      "max_seq_len": 64, "mlp_hidden": 96},
            "train": {"batch_size": 2, "grad_accum": 1, "lr": 1e-3,
                      "total_steps": 2, "warmup": 1, "save_every": 1,
                      "log_every": 1, "seed": 1337,
                      "val_frac": 0.3, "val_every": 1, "val_batches": 2},
        }
        cfg_path = os.path.join(d, "cfg.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        env = dict(os.environ, PYTHONPATH=
                   os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cmd = [sys.executable, "-m", "noesis.train", "--config", cfg_path,
               "--data", os.path.join(d, "tokens.bin"),
               "--tokenizer", os.path.join(d, "vocab.json"),
               "--out", os.path.join(d, "run")]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
        assert r.returncode == 0, f"val run failed:\n{r.stdout}\n{r.stderr}"
        assert "val split: holding out last" in r.stdout, r.stdout
        assert "[val] step 0 val_loss" in r.stdout, r.stdout      # baseline
        assert "[val] step 1 val_loss" in r.stdout, r.stdout      # periodic
        assert "[val] step 2 val_loss" in r.stdout, r.stdout
        ckpt = torch.load(os.path.join(d, "run", "latest.pt"), weights_only=False)
        assert isinstance(ckpt["best_val"], float) and math.isfinite(ckpt["best_val"])

        # ---- resume from a PRE-val checkpoint (best_val=None, v17-era format)
        mcfg = ModelConfig.from_dict(cfg["model"])
        model = NoesisModel(mcfg)
        # optimizer must mirror train.main()'s 2-group construction exactly
        decay, no_decay = [], []
        for n_, p_ in model.named_parameters():
            (no_decay if (p_.ndim == 1 or "embed" in n_) else decay).append(p_)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.1},
             {"params": no_decay, "weight_decay": 0.0}], lr=1e-3)
        save_checkpoint(os.path.join(d, "run", "latest.pt"), model, opt, 5,
                        cfg["model"], "vocab.json", "tokens.bin", 5)  # best_val=None
        cfg["train"]["total_steps"] = 6
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        r2 = subprocess.run(cmd + ["--resume"], capture_output=True, text=True,
                            env=env, timeout=900)
        assert r2.returncode == 0, f"val resume failed:\n{r2.stdout}\n{r2.stderr}"
        assert "resumed from step 5" in r2.stdout, r2.stdout
        assert "[val] step 5 val_loss" in r2.stdout, r2.stdout    # baseline after resume
        assert "[val] step 6 val_loss" in r2.stdout, r2.stdout    # first loop eval
        ckpt2 = torch.load(os.path.join(d, "run", "latest.pt"), weights_only=False)
        assert isinstance(ckpt2["best_val"], float), "best_val must be established after resume"


def test_val_eval_failure_does_not_kill_training():
    """Val split guarantee #3 (the 'Dont break anything' clause): if a val
    eval RAISES (OOM, device glitch...), the session must survive — eval is
    skipped with a WARNING, the model goes back to train mode (proven by
    training steps completing past an eval-mode-exploding forward), and the
    checkpoint carries best_val=None instead of a poisoned value."""
    import contextlib
    import io
    import noesis.model as model_mod
    import noesis.train as train_mod

    torch.manual_seed(7)
    with tempfile.TemporaryDirectory() as d:
        text_path = os.path.join(d, "corpus.txt")
        with open(text_path, "w") as f:
            f.write(SAMPLE)
        tok = BPETokenizer.train(SAMPLE, vocab_size=400)
        tok.save(os.path.join(d, "vocab.json"))
        binarize(tok, [text_path], os.path.join(d, "tokens.bin"))
        cfg = {
            "model": {"dim": 48, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
                      "head_dim": 12, "vocab_size": tok.vocab_size,
                      "max_seq_len": 64, "mlp_hidden": 96},
            "train": {"batch_size": 2, "grad_accum": 1, "lr": 1e-3,
                      "total_steps": 2, "warmup": 1, "save_every": 1,
                      "log_every": 1, "seed": 1337,
                      "val_frac": 0.3, "val_every": 1, "val_batches": 2},
        }
        cfg_path = os.path.join(d, "cfg.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        # any forward in eval mode explodes; train-mode forwards are untouched.
        # run_val's finally:model.train() is therefore observable: if it were
        # skipped after the first failure, every later train step would die too.
        orig_forward = model_mod.NoesisModel.forward

        def exploding_forward(self, *a, **k):
            if not self.training:
                raise RuntimeError("synthetic val failure")
            return orig_forward(self, *a, **k)

        model_mod.NoesisModel.forward = exploding_forward
        old_argv = sys.argv
        buf = io.StringIO()
        try:
            sys.argv = ["train", "--config", cfg_path,
                        "--data", os.path.join(d, "tokens.bin"),
                        "--tokenizer", os.path.join(d, "vocab.json"),
                        "--out", os.path.join(d, "run")]
            with contextlib.redirect_stdout(buf):
                train_mod.main()  # must NOT raise despite 3 exploding evals
        finally:
            sys.argv = old_argv
            model_mod.NoesisModel.forward = orig_forward

        out = buf.getvalue()
        assert "WARNING: eval raised RuntimeError" in out, out   # baseline + both periodic
        assert "baseline eval failed" in out, out
        assert out.count("eval failed — skipping, training continues") == 2, out
        assert "[train] step 1/2" in out and "[train] step 2/2" in out, out  # train steps ran in train mode
        ckpt = torch.load(os.path.join(d, "run", "latest.pt"), weights_only=False)
        assert ckpt["best_val"] is None, "failed evals must not poison best_val"
        assert ckpt["step"] == 2


def test_quantizer_shrinks():
    from noesis.quantize import quantize_tensor
    w = torch.randn(768, 2816)
    q, s = quantize_tensor(w)
    assert q.dtype == torch.int8
    orig = w.numel() * 4
    packed = q.nbytes + s.nbytes
    ratio = orig / packed
    assert 3.0 < ratio < 4.2, f"int8 ratio should be ~4x, got {ratio:.2f}"


def test_end_to_end_micro_train_and_generate():
    """The full pipe: corpus -> tokenizer -> binarize -> train -> sample."""
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as d:
        text_path = os.path.join(d, "corpus.txt")
        with open(text_path, "w") as f:
            f.write(SAMPLE)
        tok = BPETokenizer.train(SAMPLE, vocab_size=400)
        tok.save(os.path.join(d, "vocab.json"))
        n = binarize(tok, [text_path], os.path.join(d, "tokens.bin"))
        assert n > 1000

        cfg = ModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16,
                          vocab_size=tok.vocab_size, max_seq_len=64, mlp_hidden=128)
        model = NoesisModel(cfg)
        stream = TokenStream(os.path.join(d, "tokens.bin"), 64, 8)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        first = None
        for i in range(150):
            x, y = stream.next_batch()
            _, loss = model(x, targets=y)
            if first is None:
                first = loss.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        final = loss.item()
        assert final < first * 0.5, f"loss must fall: {first:.3f} -> {final:.3f}"
        out = generate(model, tok, "Alice was", max_new_tokens=20, temperature=0.9,
                       seed=7, device="cpu", quiet=True)
        assert isinstance(out, str) and len(out) > len("Alice was")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(fns)} smoke tests...")
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print("ALL SMOKE TESTS PASS ✅")
