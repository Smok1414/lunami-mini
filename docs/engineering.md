# Engineering notes

Things that broke during development and how they were found and fixed. Recorded here mainly because the fixes aren't obvious from reading the final code.

## NaN in the selective scan

The chunked SSD scan can be written as a cumulative-product formula (divide by an accumulated decay term to combine chunks). Under S4D-linspace initialization, that accumulated product underflows to zero within the first few steps of a chunk, and dividing by it produces `inf`, then `NaN` — on the very first forward pass, before any training happens.

Fix: the implementation in `model.py` never divides by an accumulated product. The intra-chunk recurrence is computed step by step from a zero state (bounded by construction), and the contribution carried in from a previous chunk is applied as a multiplicative decay term (which is `≤ 1` and underflows to a clean `0`, not `inf`). No cumulative product is ever materialized.

## Tokenizer design, checked against current practice first

Before writing `tokenizer_train.py`, the tokenizer configurations of Llama 3, Qwen, DeepSeek, Gemma, Mistral, SmolLM2, Phi-4, and GPT were reviewed for how each handles byte-fallback, normalization, whitespace, and digit splitting. Full reasoning and the resulting flags are in [tokenizer.md](tokenizer.md).

## A silent infinite loop in the data pipeline

`dataset.py` reserves the first `val_take` packed windows (2,000 by default) for validation and skips them when iterating the training split — correct behavior at production scale, where 2,000 windows is a negligible fraction of the data.

During a small-scale test run, the local dataset didn't contain 2,000 windows' worth of tokens. The training split iterator skipped every window it produced, reached the end of the data, and the outer training loop (which retries on `StopIteration`) simply started over — forever, without ever yielding a single training batch.

This wasn't visible in the logs (no errors, no stall messages). It was found by comparing GPU utilization (near zero) against what a data-loading-only workload should look like, then tracing back through the iteration logic. Fix: `val_take` needs to be set relative to the actual size of the dataset in use, which is now the default assumption in the small-scale test scripts.

## Throughput, measured rather than assumed

The reference Mamba-2 implementation here is plain PyTorch — no fused CUDA kernels. Measured on a T4 (GPU), context length 2048: **~330 tokens/sec**. That number, multiplied out against the available compute budget, is the basis for the honest assessment in the main [README](../README.md#benchmarks) and in [roadmap.md](roadmap.md).

## Fast Mamba backend

`model.py` can optionally use the official `mamba_ssm` package (fused Triton/CUDA kernels) instead of the reference PyTorch scan, if it's installed — same architecture, same input/output shapes, just a different internal implementation for the Mamba-2 blocks. If the import fails or the module can't be constructed, it silently falls back to the reference implementation.

This has been wired in but **not yet verified on a GPU**: attempts to `pip install mamba-ssm causal-conv1d` on the available hardware ran into a source build that took over 90 minutes to compile (the package pulls in a much larger dependency chain — TileLang, CUTLASS DSL, `quack-kernels` — than the classic version) and was abandoned before finishing. `verify_fast_mamba.py` exists to check this properly (build succeeds, forward/backward is finite, loss decreases, and a rough speed comparison against the reference path) whenever there's time to let a build run to completion.
