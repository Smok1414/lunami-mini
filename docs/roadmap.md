# Roadmap and known limitations

## Compute

The reference Mamba-2 implementation measures ~330 tokens/sec on a T4 (see [training.md](training.md)). Over a realistic full compute budget on this project, that caps out around **~90M tokens processed** — roughly 35-40x short of the ~3.3B-token Chinchilla-minimum for a model with ~165M active parameters. The architecture and training pipeline are validated end-to-end; the limiting factor right now is compute, not correctness.

Two paths forward, not mutually exclusive:

- Get the fused `mamba_ssm` backend working (see [engineering.md](engineering.md#fast-mamba-backend)) — plausible multi-x speedup on the 18 Mamba-2 layers, no architecture change.
- Access to more GPU-hours (particularly an A100, which is also required for full 8,192-token context — see below).

## Not yet done

- **Full context (8,192 tokens)** — confirmed to not fit on a T4 even at `micro_batch_size=1`. Needs an A100 (the `rocket` profile targets this, but hasn't been run yet).
- **Fast Mamba backend verification** — wired in, not yet confirmed working on a GPU (build takes a long time to compile; see [engineering.md](engineering.md)).
- **Incremental KV-cache** — `model.py`'s `use_cache` parameter is currently a stub. `chat.py` generates by recomputing the full sequence on every new token, which is correct but not efficient. A real cache (attention KV + Mamba recurrent state) is future work.
- **Chat/instruct data in the active training run** — the current run uses only the two pretrain sources (FineWeb-Edu, codeparrot-clean); Magicoder and OpenHermes are implemented in `dataset.py` but not yet included in the run in progress.
- **Downstream evaluation** — no MMLU or similar benchmark yet. At the current token count, the model is far too undertrained for this to be meaningful; see [Benchmarks](../README.md#benchmarks) for what is being tracked instead (loss, perplexity, and raw generation samples).

## Development environment

Development started on a consumer GPU (RTX 2050, 4GB) — enough to validate the full pipeline end-to-end at reduced scale, not enough for real training at the model's target size. Real training moved to Lightning.ai (T4) once the pipeline was confirmed correct.
