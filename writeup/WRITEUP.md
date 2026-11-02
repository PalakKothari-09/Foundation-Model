# minilm writeup (Questions 1–11)

Status legend: numbers marked **[GPU TODO]** need the Colab/Kaggle run and are left blank rather than guessed.
Everything else was measured on an Intel i5-1335U laptop (12 threads, no GPU), Python 3.14, PyTorch 2.15 CPU.
Raw output of the numeric answers: `writeup/analysis_output.txt` (`python scripts/analysis.py`).

## Part 1 — Warm-up answers
1. `chr(0)` is the NUL character, `'\x00'`. `repr` shows the escaped Python literal (`'\x00'`); `print` writes the raw character, which is invisible.
2. UTF-8 is compact for ASCII-heavy text and has no byte-order or fixed-width padding. `"hello"` takes 5 bytes in UTF-8, 12 in UTF-16 (BOM + 2/char) and 24 in UTF-32. For Hindi `"नमस्ते"` it is 18 / 14 / 28, so UTF-16 wins only on some scripts. UTF-8 also keeps all 256 byte values as a fixed base alphabet that can represent any string.
3. A character can span several bytes; a lone continuation byte is not valid UTF-8. `[bytes([b]).decode() for b in "नमस्ते".encode()]` raises `can't decode byte 0xe0 ... unexpected end of data`.
4. `bytes([0xC3, 0x28])`: `0xC3` starts a 2-byte character and must be followed by a continuation byte `10xxxxxx`, but `0x28` (`(`) is not one (`decode` raises; with `errors="replace"` it gives `�(`).

**Training time / memory / longest token.** Full 2.1 GB train file, 4 worker processes, 10,000-token vocab (the whole file was used, no quarter-subsampling): pre-tokenise+count **142 s**, merges **5.4 s**, total **148 s**. 59,933 unique pre-tokens. Peak memory of the main process was 105 MB (workers not included). Longest token: `b' accomplishment'` (15 bytes) — sensible, it is a long, frequent word in children's stories that gets merged all the way up with its leading space.

**Question 1 (cProfile).** Profiling the single-process run on the validation file: pre-tokenising and counting took 33.0 s of the 35.9 s total (92%) versus 2.9 s (8%) for the merge loop (my first version looped over `finditer` matches). Counting with `findall` on strings and encoding only the unique keys made pre-tokenisation ~9× faster (3.6 s vs 33 s); the merge loop (incremental pair counts + lazy heap, only touching words that contain the merged pair) never dominates. On the full file the split is 142 s vs 5.4 s, so pre-tokenising is the bottleneck and the part that benefits from multiprocessing.

## Part 2 — Encoding
- Compression ratio on 10 random validation stories: **4.138 bytes/token** (7,610 bytes → 1,839 tokens); over the full files 4.117 bytes/token (valid) and 4.116 (train).
- Speed: **3.3 MB/s** on one process (valid file, 22.5 MB in 6.9 s incl. tokenizer load); the 2.23 GB training file took **143 s** with 10 workers (15.5 MB/s aggregate) → 541,229,347 tokens, saved as `uint16` `.npy`.

**Question 2.** The vocabulary has 10,000 ids and `uint16` holds 0–65,535, so each token takes 2 bytes instead of 8 (int64): the 541M-token training file is 1.08 GB and can be memory-mapped. It stops being enough when the vocabulary exceeds 65,536 (e.g. 100k+ BPE vocabularies, multilingual models) — then `uint32` is needed.

## Part 3 — Model pieces
All pytest tests pass (`python -m pytest tests`: 52 tests; 1 is fp16-vs-fp32 and runs only when CUDA is available). They cover hand-written attention vs `F.scaled_dot_product_attention` (2-D/3-D/4-D, atol 1e-5), RoPE vs an explicit rotation-matrix version, RoPE relative-position property, shapes with extra batch dims, causality, and the parameter count.

**Question 3.** Dot products of two unit-variance `d_k`-vectors have variance `d_k`, so raw scores have std ≈ √32 = 5.66 (measured 5.81); dividing by √d_k brings it back to ≈1.10. Softmax of raw scores over 16 keys is nearly one-hot (max prob 0.987 / 0.959 / 0.805, entropy 0.08 / 0.21 / 0.70 vs 2.77 for uniform); with scaling the distribution is soft (max prob 0.32 / 0.26 / 0.22, entropy ≈2.3). A saturated softmax has tiny gradients (max |∂softmax/∂score| = 0.0125 raw vs 0.216 scaled), so unscaled attention trains poorly.

**Question 4.** With the same `q` and `k` placed at every position, `q_i·k_j` after RoPE depends only on `i−j`:
`i−j=2` → −10.142669 (i=3,j=1), −10.142670 (10,8), −10.142669 (25,23); `i−j=5` → −8.993098 (7,2), −8.993094 (30,25); `i−j=0` → −11.434472 (5,5) and (20,20), identical to the un-rotated `q·k`. Reason: the rotation by angle `iθ` on `q` and `jθ` on `k` combine to a rotation by `(i−j)θ` in every 2-D plane, since `R(a)ᵀR(b) = R(b−a)`.

## Part 4 — Model
Config: vocab 10,000, context 256, 4 layers, d_model 512, 16 heads (d_head 32), d_ff 1,344.

**Question 5.** Per layer: q,k,v,o = 4·512² = 1,048,576; SwiGLU = 3·512·1,344 = 2,064,384; 2 RMSNorm = 1,024 → 3,113,984. ×4 layers = 12,455,936. Embedding = 5,120,000, output head = 5,120,000, final norm = 512. **Total 22,696,448**, matches `sum(p.numel())` (checked in `test_param_count_of_assignment_config`). Embedding + output = 10,240,000 = **45.1%** of all parameters (transformer layers 12.46M).

**Question 6.** FLOPs (2mnp) for one forward pass over 256 tokens, 9.53 GFLOPs total:
| matmul | per layer | ×4 | share |
|---|---|---|---|
| q,k,v,o projections (4 × 256×512×512) | 0.537 G | 2.147 G | 22.5% |
| QKᵀ (16 heads × 256×32×256) | 0.067 G | 0.268 G | 2.8% |
| attention·V | 0.067 G | 0.268 G | 2.8% |
| SwiGLU w1, w3, w2 (3 × 256×512×1344) | 1.057 G | 4.228 G | 44.3% |
| output projection (256×512×10000) | — | 2.621 G | 27.5% |

The feed-forward network does the most work; the output projection is surprisingly large (27.5%) because the vocabulary is big relative to d_model.

**Question 7.** GPT-2-small-sized (12 layers, d 768, 12 heads, V 50,257, ctx 1,024, SwiGLU with d_ff 2,048 to keep the same architecture): 291.7 GFLOPs: FFN 39.8%, q/k/v/o 19.9%, attention scores+values 13.2% (QKᵀ 6.6% + AV 6.6%), output head 27.1%. As the model grows, the layer matmuls (which scale with d² per token) take over from fixed-size pieces, but attention's L² term grows with context (2.8% → 13.2% here), and the head share stays large because V grew in step with d.

## Part 5 — Training setup
**Learning-rate warm-up** (SGD on `mean(w²)`, `w = 5·randn(10,10)`, 10 steps; gradient = 2w/100, so each step multiplies `w` by `(1 − lr/50)` and the loss by its square):
- lr=1: factor 0.98 → loss 26.3 → 18.3, slow but steady decrease.
- lr=10: factor 0.8 → loss 26.3 → 0.47, fast decay.
- lr=100: factor exactly −1 → `w` flips sign every step, the loss stays at 26.3 forever (never converges).
- lr=1000: factor −19 → loss multiplies by 361 each step and diverges (2.7e24 after 10 steps).

**Tests.** Loss vs manual −log softmax; optimizer groups cover every parameter exactly once (matrices decayed, RMSNorm weights not); the schedule matches hand values at steps 0, 5, 10, 60, 110 and ends at α_min exactly at the last step; after clipping total grad norm ≤ 1; `get_batch` targets = inputs+1 with no overrun (including a dataset with a single valid window); checkpoint save → reload → same 5 losses within 1e-3 (test_checkpoint_resume_reproduces_losses). Gradients are unscaled (`scaler.unscale_`) before clipping in `train.py`. fp16-vs-fp32 loss test runs on CUDA only. **[GPU TODO: run the test file on the T4]**

**torch.compile speed-up:** **[GPU TODO: `python scripts/bench_compile.py`]**

**Question 8.** fp32 (4 bytes), 22,696,448 parameters: parameters 86.6 MiB, gradients 86.6 MiB, AdamW state (m and v) 173.2 MiB → **346.3 MiB** total (activations excluded). AdamW stores two extra tensors per parameter (the exponential moving average of the gradient and of its square), each the size of the weights, hence "twice as much again".

## Part 6 — Training script checks
- Single-batch overfit (`writeup/single_batch.png`, CPU, batch 4, lr 1e-3, 300 steps): the loss starts at 9.37 ≈ ln(10,000) = 9.21, reaches 0.024 by step 40 and 3e-5 at step 300. (Validation loss on other data is 10.9 — expected since it memorised one batch.)
- Resume check (`writeup/resume_check.png`; 60 steps, small model on CPU): stopped at step 30 and resumed with `--resume`; the loss curve is identical to the uninterrupted run (max |Δloss| = 0.0 over 60 steps). The RNG state of the batch sampler is stored in the checkpoint.
- Gradient norm over the first 100 steps (`writeup/single_batch_gradnorm.png`): 1.42 → 1.97 during warm-up, then falls steadily to 5e-4; no NaN, no growth.
- LR pilots: **[GPU TODO: `python scripts/lr_sweep.py` → `writeup/lr_sweep.png`; two sentences on how rates were chosen: I will try 1e-4, 3e-4, 1e-3, 3e-3, 1e-2 (log-spaced decade-and-a-half around typical small-transformer values 1e-3 … 3e-3), with the top rate expected to blow up.]**

**Question 9.** **[GPU TODO]**

## Part 7 — Main run
**[GPU TODO: final validation loss / perplexity, curves vs. steps and time (`scripts/plot_runs.py --time`), three stories (`scripts/sample_stories.py`), Question 10. Optional architecture experiment not done yet.]**

## Part 8 — Packaging
CPU generation speed (**Question 11**), i5-1335U, 10 torch threads, 22.7M-parameter model, 256 new tokens with no KV cache: **18.1 tokens/s** (measured with randomly initialised weights of the same shape — the compute does not depend on the weight values; re-measure with the final `release/` and `python inference.py` and update). Without a cache each step re-runs the model over the whole text so far (capped at 256): generating tokens 5…260 processes Σ min(n, 256) ≈ 33,000 token-positions to produce 256 new ones, i.e. ≈99% of the linear-layer work is repeated (the final step computes 256 positions, a KV cache computes 1). With a KV cache each step costs one position of projections/FFN plus attention over the cached keys, so the ideal saving is roughly 100× in FLOPs; real CPU speed-up will be smaller (per-step overheads, small matmuls), I would expect on the order of 5–20×. **[Measure if a KV cache is added.]**
