# Training

## Data

v1 targets English + Code only (no Russian or other languages), 50/50 mix. All sources are public — no gated access or HF token required.

| Source | Role | Weight |
|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | English pretrain | 40% |
| [codeparrot-clean-train](https://huggingface.co/datasets/codeparrot/codeparrot-clean-train) | Python pretrain | 40% |
| [Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | code instruct | 10% |
| [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | English chat | 10% |

`dataset.py` streams these directly from HuggingFace, mixes them by weight, and packs documents into fixed-length windows. Pretrain sources compute loss on every token; instruct/chat sources use ChatML formatting and mask the loss to the assistant's turn only.

The training run currently in progress uses a locally cached subset of FineWeb-Edu and codeparrot-clean rather than live streaming — HuggingFace's streaming API repeatedly stalled on shard downloads during long-running sessions (see [engineering.md](engineering.md)), and a local cache sidesteps that reliably.

## Hardware

`config.py` defines two named profiles:

| Profile | GPU | dtype | batch | grad-checkpointing |
|---|---|---|---|---|
| `workhorse` | T4 15GB | fp16 | 2×8 (eff. 16) | on |
| `rocket` | A100 40GB | bf16 | 16×2 (eff. 32) | off, `torch.compile` |

**`workhorse` is the one actually validated** — real training has run on a T4 with it. `rocket` exists in the config and hasn't been run on an A100 yet, so treat it as unverified until it has.

Confirmed on a T4: full context (8,192 tokens) does not fit in 15GB even at `micro_batch_size=1` with gradient checkpointing on — the reference (non-fused) Mamba-2 scan's memory footprint is too large at that sequence length. Training so far has used `max_seq_len=2048`, which fits comfortably (~5-8GB of the 15GB available).

## What's been run

A validation run on a T4 (context 2048, single data source, no gradient accumulation) produced this loss curve:

![Loss curve](assets/loss_curve.png)

| Step | Loss | Notes |
|---|---|---|
| 1 | 11.3868 | Matches the random-init expectation, `ln(65536) ≈ 11.09` + MoE aux loss |
| 111 | 9.4107 | |
| 112 | 9.2447 | |
| 113 | 9.0698 | |

This was enough to confirm the model, data pipeline, and optimizer are wired together correctly — loss decreases monotonically, gradient norms stay bounded, and the LR warmup schedule matches its formula exactly at every logged step.

The current, separate training run (same architecture, the FineWeb-Edu + codeparrot-clean mix described above, `max_seq_len=2048`) is in progress; see the main [README](../README.md#benchmarks) for the latest numbers pulled from it.

## Reproducing

```bash
python train.py --profile workhorse   # or rocket, once validated
python train.py --profile workhorse --resume checkpoints/step_5000.pt
```
