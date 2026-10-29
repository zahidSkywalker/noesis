"""Noesis QA smoke suite — proves every pipe before GPU money is spent.

Run:  python -m pytest tests/ -q   (or: python tests/test_smoke.py)
Covers:
  1. tokenizer round-trip (byte-level = lossless)
  2. tokenizer trained on real-ish text + encode/decode sanity
  3. model forward/backward (shapes, finite loss, grads exist)
  4. KV-cache consistency: cached generation == full-recompute greedy
  5. checkpoint save -> resume -> training continues without NaN
  6. quantizer: int8 payload ~4x smaller
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile

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
