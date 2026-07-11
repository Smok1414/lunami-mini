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

**Production run progress** (same config as above, updated as it runs). Logged every 5 steps early on to confirm stability; now sampled at coarser milestones since the loss trend is well established and a dense table stops being readable past a few hundred steps:

| Step | Loss | tok/s | Time |
|---|---|---|---|
| 5 | 11.3929 | 331 | 18:40 |
| 50 | 10.3655 | 327 | 19:18 |
| 100 | 9.5078 | 329 | 19:59 |
| 150 | 8.5769 | 330 | 20:40 |
| 300 | 6.7922 | 333 | 22:50 |
| 400 | 6.6307 | 334 | 00:11 |
| 500 | 5.7071 | 334 | 01:33 |
| 600 | 6.3581 | 334 | 02:54 |
| 690 | 5.5810 | 334 | 04:08 |

Loss is noisy step-to-step at this effective batch size (16) — e.g. step 570 briefly hit 4.6926 before bouncing back to 5.9252 the next step, and 600 (6.3581) reads worse than 500 (5.7071) despite being later — but the milestone-to-milestone trend is a clean drop from ~11.4 to the 5.7-6.4 range. Throughput crept up from ~328 to ~334 tok/s as the run progressed. Checkpoints saved every 15 steps to `checkpoints/`.

**First held-out eval, step 500:** `val_loss 6.3970 | val_ppl 600.06` — noticeably worse than the training loss at that point (5.7071 / ppl ~301). Expected this early: with ~90M realistic tokens against a ~3.3B Chinchilla-minimum, the model is still mostly memorizing local statistics of the training stream rather than generalizing, so a train/val gap this size isn't a red flag yet — just a number to watch as training continues.

Sampled `chat.py` at four points to track qualitative progress (full outputs in [Samples](../README.md#samples)):
- **Step 30** (~500K tokens): pure noise, broken tokenizer artifacts (`HashMark`, `substitutingIUS`).
- **Step 150** (~2.5M tokens): real whole words, no grammar (`situated`, `insects`, `emphasizes`).
- **Step 675** (~11M tokens): short but genuinely grammatical fragments (`This year is an important part of`) — still semantically empty, but subject-verb-object structure is starting to show up.
- **Step 3150** (~100M tokens): real Python syntax intact — `import` → `class` → docstring, in the right order. See below.

## Day 2 — Lightning.ai credits ran out, moved to Colab

Training continued past step 690 on Lightning.ai up to **step 3150** (no per-5-step log kept for this stretch — the run was left going, not actively watched). Lightning.ai's free compute ran out around there, so the plan was to resume on Google Colab instead. This surfaced two separate bugs that had never been exercised before because the run had never previously been interrupted and resumed on different hardware:

**Broke:** `torch.load(args.resume, map_location=device, ...)` loaded the full checkpoint (weights + Adam optimizer state — 2x the model size) directly onto GPU, at the same moment the freshly-constructed model and optimizer were already resident there. Two full copies in memory at once, briefly. Fine on Lightning.ai's T4 (enough headroom), but a real `CUDA out of memory` on Colab's T4 (less free memory available). Fixed by loading to CPU first and letting `load_state_dict` copy into the already-allocated GPU tensors — see [train.py](../train.py).

**Broke, the sneaky one:** resumed cleanly after that fix, got through data loading, then died with `CUDA out of memory: Tried to allocate 4.00 GiB` inside `F.cross_entropy`. `ModelConfig.max_seq_len` defaults to 8192 in `config.py`, and neither hardware profile (`rocket`/`workhorse`) overrides it — so a fresh clone on Colab silently trained at full 8192 context, which [roadmap.md](roadmap.md#compute) already documents as not fitting a T4 even at `micro_batch_size=1`. The math checks out exactly: `2 × 8192 × 65536 × 4 bytes = 4.00 GiB`, the exact allocation the error reported. Every checkpoint so far was actually trained at `max_seq_len=2048`; that just wasn't enforceable from the CLI. Added a `--max-seq-len` flag so this doesn't require hand-editing `config.py` again.

**Sample, step 3150.pt** (~100M tokens at ctx=2048 — first checkpoint tested past the ~90M-token "honest" budget from Day 1, and still going):

> **Prompt:** `Hello, how are you?`
> **Output:** `Each person? Should you?  """  # get the main module # from __future__ import print_function  import re  from ansible.module_utils.six import AnsibleModule  class types(object):     """     A version class for`

Still not a coherent reply, but a clear step up from step 675: two well-formed (if non-sequitur) questions, followed by genuinely correct Python structure — `from __future__ import ...`, `import re`, then a class definition with a docstring, in the order real Python files actually use them. First clear sign the `codeparrot-clean` half of the data mix is landing.
