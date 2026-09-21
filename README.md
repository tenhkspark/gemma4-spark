[English](README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [中文](README.zh.md)

Gemma 4 26B A4B on a single DGX Spark.
Official NVFP4 as-is: 28.8 tok/s single-stream.
This recipe: 109.7 tok/s single-stream, 1,080.8 tok/s aggregate at 32 concurrent.

# gemma4-spark — Gemma 4 26B A4B NVFP4 (untied lm_head) on NVIDIA DGX Spark

## What this is

A configuration we call **A+γ8**:

- Built on NVIDIA's official NVFP4 checkpoint (`nvidia/Gemma-4-26B-A4B-NVFP4`)
- `tie_word_embeddings` removed and **lm_head also quantized to NVFP4**
  (appendix `untie-lmhead-fp8.py` shows how this weight was made)
- MTP speculative decoding, `num_speculative_tokens = 8` (γ8), draft
  `google/gemma-4-26B-A4B-it-assistant`
- `--language-model-only` (the vision tower is not loaded)
- FP8 KV cache

## Contents

- `serve.sh` / `gemma4.env` / `gemma4.small.env` — one-node serve
  harness (`up` / `down` / `status` / `smoke`)
- `BRING-UP.md` — full bring-up recipe (Japanese)
- `SERVING-NOTES-2026-09-20.{ja,ko,zh}.md` — the serving record; ko/zh
  are translated from the final Japanese version
- `MODEL-CARD.md` — the model card
- `untie-lmhead-fp8.py` — appendix: how this weight was built
- `bench-cell.py` — one C=1 bench cell for bring-up verification
- `LICENSE`, `NOTICE`, `LICENSE-CHECK.md`, `check.sh`, `upload.sh`

## Requirements

- One DGX Spark (GB10, 128 GB unified memory), or a Blackwell-
  generation GPU (NVFP4 compute needs that generation)
- About 50 GB of disk (weights 19.2 GB + draft 0.8 GB + container
  image ~30 GB)
- The container image — `docker pull tenhkspark/vllm-gb10:v0.28.0-sm121`
  (9.4 GB compressed). To build it yourself: `BRING-UP.md` §2

## Quickstart

1. **Fetch the weights** —
   `huggingface-cli download tenhkspark/gemma-4-26B-A4B-NVFP4-lmhead --local-dir ./gemma4-lmhead`.
   Manifest: 13 files / 19,240,726,248 B / md5
   `571932348835310ce77799f70a4e9814`.
2. **Fetch the container** — `docker pull tenhkspark/vllm-gb10:v0.28.0-sm121`.
   To build it yourself instead, see `BRING-UP.md` §2.
3. **`./serve.sh up`** — on a ~32 GB discrete GPU,
   `./serve.sh up --env gemma4.small.env`.

Smoke check: `./serve.sh smoke` (one Japanese question, prints tok/s).

## Measured results (DGX Spark, 2026-09-20)

### Speed

Practical Japanese business prompts (3 shapes × 30 documents × 2
repetitions, temperature 0, headline tok/s including TTFT):

| single request (C=1) | tok/s |
|---|---:|
| official NVFP4 as-is, speculation off | 28.8 |
| official NVFP4 as-is, γ8 | 100.5 |
| A+γ8 (this recipe) | **109.7** |
| same, speculation off | 35.8 |

Speculative decoding speedup: **3.06×** (109.7 / 35.8); on the
unmodified official checkpoint it is 28.8 → 100.5, **3.49×**.
Quantising lm_head adds 100.5 → 109.7, **+9.2%** (with speculation
off, 28.8 → 35.8, +24.3%). Aggregate throughput under
steady load (60 s per level):

| concurrency | tok/s |
|---:|---:|
| C=8 | 496.7 |
| C=16 | 822.0 |
| C=32 | 1,080.8 |

### Practical tasks (90 Japanese tasks, temperature 0)

| concurrency | passed | tasks/min |
|---:|---:|---:|
| C=1 | 84/90 | 7.59 |
| C=8 | 82/90 | 35.90 |
| C=32 | 80/90 | 83.40 |

Per category: **finance (pulling figures out of a table) is 30/30 in every
condition**; monitor is 30/30 at C=1 (answering "action needed" every time
scores 15/30). db is the weakest — almost every failure is naming an extra
table. The same tendency shows up in BF16 without quantisation (20/30 at C=1
against 24/30 for this recipe), so it is not caused by quantisation.

Measure with **prefix caching off**: confirm `enable_prefix_caching=False` in
the startup log first. With it on, resending the same prompts lets prefill come
back from cache. At 2,048 in / 32 out, a second pass over the same prompts ran
more than 20x faster — that figure is not real throughput, so it is not in the
tables above.

### Long input

| input length | C=1 (tok/s) | C=8 aggregate (tok/s) |
|---:|---:|---:|
| 8K | 30.6 | 62.9 |
| 28K | 10.4 | 12.0 |

Prefill-bound, not KV-bound — split long inputs.

### Quality (paired diff vs BF16, pt)

JGLUE valid + JMMLU; n = 2,434 per metric (JCommonsenseQA: all 1,119).

| metric | Δ (pt) | one-sided 95% lower bound |
|---|---:|---:|
| JSQuAD EM | −0.66 | −1.17 |
| JSQuAD char-F1 | −0.17 | −0.41 |
| JNLI acc (run 1) | −1.23 | −1.93 |
| JNLI acc (run 2) | −1.48 | −2.14 |
| JCommonsenseQA acc | −0.36 | −1.07 |
| JMMLU acc | −0.70 | −1.60 |

**All five metrics land within −2 pt of BF16 as point estimates. For
JNLI the lower bound straddles −2 pt, so statistical non-inferiority
is not confirmed.**

Independent set — a holdout of 500 questions per task that was not
used for configuration selection:

| metric | Δ (pt) |
|---|---:|
| JSQuAD EM | +0.20 |
| JSQuAD char-F1 | −0.18 |
| JNLI acc | −1.60 |
| JCommonsenseQA acc | −0.20 |

JMMLU is excluded: its structure has no train/valid split, so no
holdout can be built.

### Memory reservation

DGX Spark is unified memory: the GPU and the OS share the same 128 GB.
`--gpu-memory-utilization` is the fraction vLLM reserves at startup,
held regardless of load. The default is **0.5** (`GEMMA4_GPU_UTIL` in
`gemma4.env`): the startup log reports `GPU KV cache size: 607,998
tokens` and the OS keeps 51-52 GB free. Raising util grows the KV pool
but starves the OS — at 0.9 the OS side dropped to a few GB and a
co-located process was killed (measured).

| GPU_UTIL | KV tokens | 32K windows | OS headroom |
|---|---:|---:|---:|
| 0.5 (default) | 607,998 | ~18.6 | 51-52 GB |
| 0.6 | 837,957 | ~25.6 | 40 GB |
| 0.7 | 1,011,401 | ~30.9 | 27 GB |
| 0.9 | — | — | a few GB; a co-located process was killed |

The speed and quality tables above were measured at util 0.6. The 0.5
default stays consistent with them: single-request speed does not
depend on KV pool size (0.5 measured 104.5 tok/s, within noise), and
the C=32 workload needs only about 50K KV tokens against the 607,998
available at 0.5.

### Size

Distribution payload ≈ 19 GB. On-GPU footprint measured in the startup
log: 17.08 GiB (`Model loading took`).

## Other GPUs (estimates, not measured)

Everything in this section is an estimate; measurements were taken on
DGX Spark only. NVFP4 compute needs a Blackwell-generation GPU.

- A discrete GPU does not share VRAM with the OS, so util can be raised
  to 0.85-0.9.
- Budget breakdown: 17.08 GiB weights (measured from the startup log) + ~0.8 GB MTP draft + KV pool.
- 64 GB GPU: at `--max-model-len 32768`, expect a KV pool of about
  35 GB (~20 concurrent 32K windows).
- 32 GB GPU: start from `--max-model-len 8192` and `--max-num-seqs 8`
  — that is exactly what `gemma4.small.env` carries.

Without `serve.sh`, the assembled command is (gemma4.small.env
profile; `/checkpoint` and `/checkpoint-mtp` are the mounted weights
and draft dirs):

```bash
vllm serve /checkpoint --served-model-name gemma4 --host 0.0.0.0 --port 8890 --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 8192 --enable-chunked-prefill --no-enable-prefix-caching --language-model-only --trust-remote-code --reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice --limit-mm-per-prompt '{"image":0,"audio":0}' --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 --speculative-config '{"method":"mtp","model":"/checkpoint-mtp","num_speculative_tokens":8}'
```

## Limitations

- Long inputs are prefill-bound; split them (table above).
- Even at temperature 0 the output diverges slightly between machines
  (the weights are identical; the serving side differs).
- In the practical-task eval, the db task (extracting table names from
  SQL) tends to list extra tables. The same tendency appears with BF16
  (BF16 20/30 vs this recipe 24/30 at C=1), so it is not caused by
  quantization.
- The API has no authentication: the server listens on all interfaces.
  Run it on a trusted network or bind to localhost.

## Next steps

### All-layer 4-bit with Japanese calibration (tried, not adopted)

The weights in this recipe keep attention and the shared MLP in BF16. We also built
a variant that quantises those to NVFP4 as well, using 365 Japanese calibration
samples and error-compensated quantisation (GPTQ family), and measured it.

| metric | this recipe | all-layer 4-bit |
|---|---:|---:|
| single-stream tok/s | 109.7 | **122.3 (+11.5%)** |
| aggregate tok/s at C=32 | 1,080.8 | **586.5 (about half)** |
| JNLI paired delta | -1.23 / -1.48 pt | -1.64 pt |
| all 5 metrics, point estimate | within -2 pt | within -2 pt |

**Quality held, but throughput under concurrency dropped sharply, so we did not
adopt it.** Single-stream is faster and Japanese generation stays sound. We do not
know why only the concurrent case is slow (single-stream is bandwidth-bound while
concurrency moves into a compute-bound regime — but that is unverified).

If the concurrency drop can be explained and avoided with the same measurement
method, the +11.5% single-stream gain may be reachable without losing throughput.

## License

Apache License 2.0 — see `LICENSE` and `NOTICE`. Use of the model
itself remains subject to the Gemma Terms of Use and its
prohibited-use policy.

## Tools used

Tools used — Claude Fable 5.1, GPT-6 (Astra), GLM-5.3-Flash. All code in this repository was written for this project.

## Acknowledgements

Google (Gemma 4 and the MTP draft), NVIDIA (the NVFP4 checkpoint and
DGX Spark), and vLLM (the serving engine).
