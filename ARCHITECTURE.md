# NOESIS — Architecture

## The brain (125M-class)

| Component | Choice | Why |
|-----------|--------|-----|
| Tokenizer | Own byte-level BPE, 32,768 vocab | Lossless, ours; 32k (vs GPT-2's 50k) frees ~13M params for thinking |
| Position | RoPE | Long-context friendly, no learned position table |
| Norm | RMSNorm, pre-norm, fp32 | Stable, fast, modern |
| Attention | GQA: 12 query heads, 8 KV heads, head_dim 64 | ~80% of MHA quality, cheaper KV-cache |
| MLP | SwiGLU, hidden 2816 | Best quality/param tradeoff |
| Embeddings | Tied input/output | Saves 25M params at 32k vocab |
| Params | **~126.6M** | 25.2M emb + 8.45M x 12 layers |
| Context | 1024 | Quota-efficient; upgradable later |

## The 5 layers of the machine

1. **Mouth & Ears** — own BPE tokenizer + document pipeline (PDF/DOCX/TXT/MD -> clean -> chunk)
2. **The Brain** — this repo's transformer, trained by next-token prediction
3. **Self-Digest** — (a) own embedding model + local vector store (fast recall),
   (b) docs auto-distilled into QA pairs -> LoRA micro-training (knowledge becomes weights)
4. **Inference Engine** — hand-rolled KV-cache generation + int8 quantizer (~150MB brain)
5. **Sleep Cycle** — nightly self-training on conversations + feedback; eval gate; auto-rollback

## Training discipline (single-T4 sessions)

- Checkpoint every 500 steps: model + optimizer + scheduler + step -> versioned dataset snapshots
- Sessions are disposable; progress is not
- Session budget: ~9h wall each, chained head-to-tail for the full ~2.5B-token education
- bf16 autocast on T4; grad-accum 16x for effective batch 131k tokens
