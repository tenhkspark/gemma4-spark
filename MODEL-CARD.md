---
license: apache-2.0
base_model:
  - google/gemma-4-26B-A4B-it
  - nvidia/Gemma-4-26B-A4B-NVFP4
language:
  - ja
tags:
  - gemma4
  - nvfp4
  - vllm
---

# Gemma 4 26B A4B — NVFP4 with untied lm_head

[English](README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [中文](README.zh.md)

Gemma 4 26B A4B on a single DGX Spark.
Official NVFP4 as-is: 28.8 tok/s single-stream.
This recipe: 109.7 tok/s single-stream, 1,080.8 tok/s aggregate at 32 concurrent.

NVFP4 derivative checkpoint built on `nvidia/Gemma-4-26B-A4B-NVFP4` —
NVIDIA's official ModelOpt NVFP4 quantization of Google's
`gemma-4-26B-A4B-it` (BF16). The stock checkpoint ties `lm_head` to the
BF16 embedding; this derivative re-saves it with
`tie_word_embeddings=false` and a separately NVFP4-quantized
`lm_head.weight`, removing the tied embedding's per-token read cost.

Weights: 19.2 GB. Serving was validated with vLLM on a single NVIDIA
DGX Spark (GB10).

## Quickstart

1. **Fetch the weights** —
   `huggingface-cli download tenhkspark/gemma-4-26B-A4B-NVFP4-lmhead --local-dir ./gemma4-lmhead`.
   Manifest to verify against: 13 files / 19,240,726,248 B / md5
   `571932348835310ce77799f70a4e9814`.
2. **Fetch the container** — `docker pull tenhkspark/vllm-gb10:v0.28.0-sm121`
   (9.4 GB compressed). Or build vLLM's upstream `docker/Dockerfile`
   (build command and args: see `BRING-UP.md` §2).
3. **`./serve.sh up`** — for a ~32 GB discrete GPU,
   `./serve.sh up --env gemma4.small.env` instead.

Smoke check: `./serve.sh smoke` sends one Japanese question and prints
tok/s.

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

### Memory reservation (DGX Spark unified memory)

| GPU_UTIL | KV tokens |
|---|---:|
| 0.5 (default) | 607,998 |
| 0.6 | 837,957 |
| 0.7 | 1,011,401 |

### Size

Distribution payload ≈ 19 GB. On-GPU footprint measured in the startup
log: 17.08 GiB (`Model loading took`).

## Without serve.sh (other GPUs)

`serve.sh` assembles and runs this command (gemma4.small.env profile;
`/checkpoint` and `/checkpoint-mtp` are the mounted weights and draft
dirs):

```bash
vllm serve /checkpoint --served-model-name gemma4 --host 0.0.0.0 --port 8890 --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 8192 --enable-chunked-prefill --no-enable-prefix-caching --language-model-only --trust-remote-code --reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice --limit-mm-per-prompt '{"image":0,"audio":0}' --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 --speculative-config '{"method":"mtp","model":"/checkpoint-mtp","num_speculative_tokens":8}'
```

## Limitations

- Throughput is prefill-bound. At 8K input tokens the single-stream
  rate drops to about a third of the short-input figure, and at 28K to
  about a tenth. Split long inputs instead of feeding them whole.
- Generation is not bit-reproducible across machines: even at
  temperature 0 the output diverges slightly between hosts. The weights
  are identical; the divergence is on the serving side.
- In the practical-task eval, the db task (extracting table names from
  SQL) tends to list extra tables. The same tendency appears with
  BF16 weights (BF16 20/30 vs this recipe 24/30 at C=1), so it is not
  caused by quantization.
- The API has no authentication: `--host 0.0.0.0` answers on every
  interface. Run it on a trusted network or bind to 127.0.0.1.

## Distribution format

To be finalized at release time.

### Keeping attention in FP8 (if quality matters most)

A variant with attention (QKVO) left in FP8 and the shared MLP, routed
experts and lm_head in NVFP4, calibrated on the same 365 Japanese
samples: **117.5 tok/s** single-stream (+7.1%), JNLI paired delta
**-0.74 pt** on the validation set and **-1.20 pt** on the held-out set
— both better than this recipe. Aggregate throughput at 32 concurrent
is about half, so it is not the published configuration; batch work
needs the throughput. Weights are not published. Raising attention
precision recovers quality and costs a little single-stream speed
(against the all-4bit variant: JNLI -1.64 → -0.74 pt, 122.3 → 117.5
tok/s).


## Why lm_head

A decode step took 34.775 ms, so we profiled it with the torch profiler.

| kernel | ms/step | share |
|---|---:|---:|
| cuBLAS BF16 GEMV | 28.64 | 82.5% |
| MoE (routing + expert GEMM) | 4.75 | 13.6% |
| attention (Triton) | 0.643 | 1.85% |

The top 15 kernels account for 99.1% of the step. Most of the time goes
into reading linear layers that are still BF16.

NVIDIA's official NVFP4 quantizes the routed experts only; its
`config.json` lists 93 entries under `quantization_config.ignore`
(`mlp*` / `router*` / `self_attn*` for all 30 layers, plus `lm_head` and
the vision stack). Summing the safetensors headers, 5.35 GB of BF16 is
read per token: QKVO projections 2.51 GB, lm_head (tied embedding)
1.48 GB, shared-expert dense MLP 1.34 GB, router and norms 0.02 GB.

5.35 GB / 28.64 ms is 186.8 GB/s effective, 68.4% of the GB10's
273 GB/s — the volume read, not kernel efficiency.

Of those, lm_head can be quantized on its own by untying it from the
embedding. At the same γ8 that takes 100.5 → **109.7 tok/s** (+9.2%);
with speculation off, 28.8 → **35.8 tok/s** (+24.3%).

What did not help: forcing the MoE kernel to MARLIN left non-speculative
single-stream unchanged; adding `--quantization fp8` on top of the NVFP4
checkpoint is ignored (the startup log still reads
`quantization=modelopt_fp4`); the vision tower (0.59 GB) is not read
during text-only decode, and `--language-model-only` did not change the
speed.

## License

Apache License 2.0 — see `LICENSE`. This is a derivative checkpoint;
`NOTICE` records what was changed. Use of the model itself remains
subject to the Gemma Terms of Use and its prohibited-use policy.

## Tools used

Tools used — Claude Fable 5.1, GPT-6 (Astra), GLM-5.3-Flash. All code in this repository was written for this project.
