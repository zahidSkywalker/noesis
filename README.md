# NOESIS

> **Think from scratch.**

A ground-up language model. Every layer of intelligence — tokenizer, architecture,
weights, quantizer, inference engine — is built here, in this repo, from zero.

- **Zero** pretrained models
- **Zero** external AI APIs
- **Zero** API keys

## Repository layout

```
noesis/
├── noesis/
│   ├── tokenizer.py    # byte-level BPE, trained from scratch (lossless round-trip)
│   ├── model.py        # decoder-only transformer: RoPE, RMSNorm, SwiGLU, GQA, KV-cache
│   ├── data.py         # corpus -> uint16 token stream -> batches
│   ├── train.py        # pretraining loop: AdamW, cosine schedule, grad-accum, cold+warm checkpoints
│   ├── generate.py     # streaming inference: temperature, top-p, repetition penalty
│   └── quantize.py     # weight-only int8 (per-channel) -> portable npz
├── configs/
│   ├── nano.json       # ~1M-param QA config (CPU, minutes)
│   └── base_125m.json  # the real 125M brain (single T4)
└── tests/
    └── test_smoke.py   # QA suite: lossless tokenizer, cache==recompute, resume, quantization
```

## Quickstart (QA smoke, CPU)

```bash
pip install -r requirements.txt
python tests/test_smoke.py
```

## Train

```bash
# nano (CPU, minutes)
python -m noesis.prepare --corpus corpus/*.txt --vocab 2048 --out runs/nano   # (M3 tooling)
python -m noesis.train  --config configs/nano.json --data runs/nano/tokens.bin \
                        --tokenizer runs/nano/vocab.json --out runs/nano
# the 125M brain (single T4)
python -m noesis.train --config configs/base_125m.json --data ... --tokenizer ... --out runs/base --resume
```

## Generate

```bash
python -m noesis.generate --run runs/nano --prompt "Once upon a time" --tokens 200
```

## The doctrine

1. **From scratch means from scratch.** No borrowed tokenizer, no borrowed weights.
2. **Knowledge is digested, not just retrieved.** Documents become training signal.
3. **Real hardware.** Trains on a single T4 (4 vCPU, ~30GB RAM); inference runs on 4 CPU cores.
4. **Portable brain.** ~150MB quantized — runs on a laptop, in a browser, anywhere.
