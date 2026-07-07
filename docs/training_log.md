# Training log

Running notes from actual development sessions — what broke, what got fixed, what the numbers were. Written as it happened, not cleaned up afterward.

## Day 1

**Built:** `config.py`, `model.py` (Mamba-2 + GQA + MoE), `tokenizer_train.py`, `dataset.py`, `train.py`, `chat.py`.

**Tokenizer:** Trained on a 1.1GB EN+Code corpus (576.8MB FineWeb-Edu, 576.8MB codeparrot-clean) streamed live from HuggingFace — this part worked without issue, ~20MB/s, done in under a minute of actual download time. See [tokenizer.md](tokenizer.md) for the design decisions checked before training it.

**Broke:** `SentencePiece` training briefly stalled repeatedly — traced to log noise from `tqdm` progress bars mixing with SentencePiece's own C++ logging. Not a real bug, just noisy output that looked like a hang.

**Broke, actually:** The chunkwise selective scan produced `NaN` on the very first forward pass with S4D-linspace initialization. Root cause and fix in [engineering.md](engineering.md#nan-in-the-selective-scan).

**Moved to Lightning.ai (T4)** once the pipeline was validated locally on a 4GB consumer GPU at reduced scale. Full-scale model (165M active) loads and runs on a T4 without issue.

**Broke:** Full context (8,192 tokens) ran out of memory on the T4 even at `micro_batch_size=1`. Not a bug — the reference (non-fused) Mamba-2 scan's memory footprint is genuinely too large at that sequence length on 15GB. Switched to `max_seq_len=2048` for validation, which fits with room to spare.

**Validation run (ctx=2048, single data source):**

| Step | Loss |
|---|---|
| 1 | 11.3868 |
| 111 | 9.4107 |
| 112 | 9.2447 |
| 113 | 9.0698 |

Loss decreases monotonically, gradient norms stay bounded, LR warmup matches its formula at every step. This is the run that confirmed the whole pipeline is wired correctly end-to-end.

**Broke, the expensive one:** Live HuggingFace streaming (with shuffling) repeatedly stalled for long periods with no error — looked exactly like a hang. Root cause turned out to be in this project's own code: `dataset.py` reserves the first 2,000 packed windows for validation and skips them for training, which is fine at production scale but meant a small test dataset never produced a single real training batch — full explanation in [engineering.md](engineering.md#a-silent-infinite-loop-in-the-data-pipeline). Switched to a locally cached data subset to avoid the live-streaming stalls entirely.

**Attempted:** Installing `mamba_ssm` for a fused-kernel backend. `pip install --no-build-isolation mamba-ssm causal-conv1d` triggered a from-source build that ran for 90+ minutes (confirmed actively compiling via `top`, not hung) and was abandoned to keep moving. Wired into `model.py` with automatic fallback regardless — see [engineering.md](engineering.md#fast-mamba-backend). Revisiting this is on the [roadmap](roadmap.md).

**Real training started:** full architecture (165M active), `max_seq_len=2048`, FineWeb-Edu + codeparrot-clean mix, `workhorse` profile (T4, fp16). Measured throughput: **~330 tokens/sec**. Checkpoint at step 30 loaded correctly in `chat.py` and generated text (incoherent, as expected at ~500K tokens of training — see [Samples](../README.md#samples) in the main README).

**Honest number for the day:** at ~330 tok/s, the realistic compute budget available for this project caps out around 90M tokens total — well short of what a 165M-active-parameter model would need for genuinely coherent output. Full reasoning in [roadmap.md](roadmap.md#compute).
