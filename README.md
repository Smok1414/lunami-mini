# Lunami-Mini

An independent, from-scratch LLM — not a fine-tune, not a wrapper around an existing model. A single, focused **English + Code** model: hybrid Mamba-2 + GQA-Transformer + Mixture-of-Experts, ~165-170M active parameters.

## Architecture

- 24 layers: `[Mamba, Mamba, Mamba, Attention] × 6` (75% Mamba-2 / 25% GQA attention)
- `d_model=768`, 8192-token context window
- **Mamba-2**: state dim 64, expand 2, conv kernel 4 — chunkwise selective scan implemented in pure PyTorch (no `mamba_ssm` dependency)
- **GQA attention**: 8 query heads / 2 KV heads, head_dim 96, RoPE, FlashAttention-2 (via `torch.scaled_dot_product_attention`)
- **FFN**: Mixture-of-Experts — 8 SwiGLU experts, top-2 routing, load-balancing aux loss
- RMSNorm, tied embeddings
- Vocabulary: 65,536-token SentencePiece BPE (byte-fallback, ChatML special tokens)
- ~165-170M active parameters/token, ~293M total stored (due to MoE)

## Project structure

| File | Role |
|---|---|
| `config.py` | single source of truth: hyperparameters, special tokens, datasets, hardware profiles |
| `model.py` | LunamiLM architecture |
| `tokenizer_train.py` | SentencePiece BPE tokenizer training |
| `dataset.py` | HuggingFace dataset streaming, mixing, packing, ChatML loss masking |
| `train.py` | training loop (AMP, gradient accumulation, checkpoints, resume) |
| `chat.py` | inference / REPL chat |

## Data

v1 is **English + Code only** — no Russian or other languages. 50/50 mix:

| Source | Role | Weight |
|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | English pretrain | 40% |
| [codeparrot-clean-train](https://huggingface.co/datasets/codeparrot/codeparrot-clean-train) | Python pretrain | 40% |
| [Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | code instruct | 10% |
| [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | English chat | 10% |

All sources are public — no gated access or HF token required (see `config.default_dataset_sources()`).

## Hardware profiles

Built for a hybrid Lightning.ai Studios workflow — switch with a single flag:

| Profile | GPU | dtype | batch | notes |
|---|---|---|---|---|
| `rocket` | A100 40GB | bf16 | 16×2 (eff. 32) | `torch.compile`, no grad-checkpointing |
| `workhorse` | T4 15GB | fp16 | 2×8 (eff. 16) | GradScaler, gradient checkpointing |

```bash
python train.py --profile rocket      # A100
python train.py --profile workhorse   # T4
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. **Train the tokenizer** — put English text and code files into `tokenizer_corpus/`, then:
   ```bash
   python tokenizer_train.py
   ```
   Produces `tokenizer/tokenizer.model` + `tokenizer/tokenizer.vocab`.

2. **Train the model** (data streams directly from HuggingFace, no manual download needed):
   ```bash
   python train.py --profile workhorse   # or rocket
   ```
   Resume training: `python train.py --profile workhorse --resume checkpoints/step_5000.pt`

3. **Chat with the model**:
   ```bash
   python chat.py --checkpoint checkpoints/step_50000.pt
   ```

## Project status

- [x] Architecture implemented and validated live (real forward/backward passes, finite decreasing loss)
- [x] Tokenizer trained on a real 1.1GB EN+Code corpus, 65,536 vocab, round-trip verified on code/emoji/Unicode
- [x] Training loop validated end-to-end on real streamed data + checkpoint resume (T4, reduced context 2048: loss 11.39→9.07 over 113 real steps)
- [ ] Full-scale training run at ctx=8192 (needs A100 — confirmed T4 OOMs even at micro_batch=1 at full context)
- [ ] Fast Mamba backend (`mamba_ssm`) integrated but not yet verified on GPU (see `verify_fast_mamba.py`)

## Performance note

The reference Mamba-2 implementation is pure PyTorch (no fused CUDA kernels) — measured throughput on T4 at ctx=2048 was **~164 tok/s**, far too slow for a real pretraining run (even the full ~77 T4-hours a small compute budget buys would only cover ~45M tokens, well short of the ~3.3B-token Chinchilla-minimum for this parameter count). `model.py` now optionally uses the official `mamba_ssm` fused backend when installed (`pip install mamba-ssm causal-conv1d`, Linux+CUDA only) — same architecture, just a faster internal implementation for the Mamba-2 blocks. **Run `python verify_fast_mamba.py` on a GPU box before trusting it for real training** — it hasn't been verified on hardware that can actually compile `mamba_ssm` yet.

## Known limitations

- `model.py` does not implement an incremental KV-cache (the `use_cache` parameter is a stub) — generation in `chat.py` uses honest full-recompute. A real cache is future work, not part of v1.
- Development was done on a consumer GPU (RTX 2050, 4GB) — enough to validate the full pipeline (tokenizer + data + training), but not enough for a full pretraining run at the model's target scale.

## License

MIT — see [LICENSE](LICENSE).
