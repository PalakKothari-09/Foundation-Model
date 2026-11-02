# minilm — a small GPT-style language model from scratch

Byte-level BPE tokenizer, RoPE/SwiGLU/RMSNorm Transformer, training loop and sampler, all written by hand
(PyTorch primitives only) and trained on TinyStories (22.7M parameters, 10k vocab, context 256).

## Generate a story (CPU only, no training code or data needed)

```bash
pip install -r requirements.txt
python inference.py --model-dir release --prompt "Once upon a time" \
    --max-new-tokens 256 --temperature 0.8 --top-p 0.9 --seed 42 --device cpu
```

Same `--seed` gives the same story. The last line reports tokens generated and tokens/s.

## Layout

```
minilm/tokenizer/   bpe.py (training), tokenizer.py (Tokenizer class, save/load)
minilm/model/       layers.py (RoPE, attention, SwiGLU), transformer.py (block, TransformerLM)
minilm/train/       core.py (loss, AdamW groups, cosine schedule, clipping, batches, checkpoints)
minilm/generate.py  temperature + top-p sampling
train.py            training script (all settings are CLI flags / --config JSON; --resume)
inference.py        loads release/ and generates
scripts/            train_tokenizer, encode_dataset, lr_sweep, bench_compile, resume_check,
                    plot_runs, sample_stories, make_release, analysis
tests/              pytest suite   ->   python -m pytest tests
tokenizer_files/    the trained 10k vocab + merges
release/            model.pt, config.json, vocab.json, merges.txt
notebooks/          thin Colab driver (all logic lives in the .py files)
writeup/            answers to Questions 1-11, plots
```

## Reproduce

```bash
pip install -r requirements-dev.txt
python -m pytest tests
python scripts/train_tokenizer.py --input TinyStoriesV2-GPT4-train.txt --out tokenizer_files --workers 4
python scripts/encode_dataset.py --input TinyStoriesV2-GPT4-train.txt --out data/train.npy --workers 4
python scripts/encode_dataset.py --input TinyStoriesV2-GPT4-valid.txt --out data/valid.npy
python scripts/lr_sweep.py                      # 5 pilot runs x 300 steps
python train.py --lr <best> --max-steps 5000 --out-dir runs/main
python scripts/make_release.py --ckpt runs/main/ckpt.pt --out release
```
