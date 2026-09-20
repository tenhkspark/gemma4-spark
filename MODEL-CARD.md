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

NVFP4 derivative checkpoint built on `nvidia/Gemma-4-26B-A4B-NVFP4` —
NVIDIA's official ModelOpt NVFP4 quantization of Google's
`gemma-4-26B-A4B-it` (BF16). The stock checkpoint ties `lm_head` to the
BF16 embedding; this derivative re-saves it with
`tie_word_embeddings=false` and a separately NVFP4-quantized
`lm_head.weight`, removing the tied embedding's per-token read cost.

Weights: 19.2 GB. Serving was validated with vLLM on a single NVIDIA
DGX Spark (GB10).

## Quickstart

1. **Fetch the weights.** The distribution format will be fixed at
   release; the manifest to verify against is 13 files /
   19,240,726,248 B / md5 `571932348835310ce77799f70a4e9814`.
2. **Prepare the container.** Build vLLM's upstream `docker/Dockerfile`
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
| A+γ8 (this recipe) | **109.7** |
| same, speculation off | 35.8 |

Speculative decoding speedup: **2.91×**. Aggregate throughput under
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

## License

Apache License 2.0 — see `LICENSE`. This is a derivative checkpoint;
`NOTICE` records what was changed. Use of the model itself remains
subject to the Gemma Terms of Use and its prohibited-use policy.

## Tools used

Tools used — Claude Fable 5.1, GPT-6 (Astra), GLM-5.3-Flash. All code in this repository was written for this project.
