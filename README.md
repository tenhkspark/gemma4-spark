# Gemma 4 26B-A4B on DGX Spark, v2

Serving setup for the NVFP4 build of Gemma 4 26B-A4B on a single DGX Spark. The model weights are unchanged from v1. v1: git tag `v1`.

## v1 to v2

| tok/s unless noted | v1 | v2 |
|---|---|---|
| Context length | 32,768 | 262,144 |
| Time to first token, 32k prompt | 12.4 s | 6.1 s (prefill-first profile) |
| Time to first token, 128k prompt | not supported | 56 s (prefill-first profile) |
| Time to first token, 250k prompt | not supported | 182 s (prefill-first profile) |
| Decode, Japanese chat, tok/s at concurrency 1 / 32 | 45.9 / 616 | 62.9 / 850 |
| Decode, coding | 67.8 / 644 | 69.5 / 725 |
| Decode, tool calling | 54.6 / 193 | 61.4 / 272 |
| Decode, long documents | 29.1 / 152 | 38.0 / 448 |
| Decode, structured extraction (`GEMMA4_MTP=8` option) | 109.7 / 1,080.8 (as published) | about 100 / 934 |
| Quality (benchmark and real-use checks) | baseline | parity |
| Cold start | about 4 min | about 4 min |

Time to first token at 2k / 8k / 30k prompts is the same as v1 in the balanced profile (0.32 s / 1.57 s / 12.3 s). Stability: 15 minutes at 32 short plus 4 long (up to 131k) concurrent requests through the router, 0 errors, 0 timeouts.

## What to change when upgrading

- Image: [tenhkspark/gemma-4-v2:v2](https://hub.docker.com/r/tenhkspark/gemma-4-v2)
- Env files: `gemma4-v2.env` plus one profile, `gemma4-v2-balanced.env` (default) or `gemma4-v2-prefill-first.env` (long prompts)
- Serve script: `gemma4-v2-serve.sh`, chat template `chat_template.jinja`
- Router (optional, several nodes): `tools/router.py` with `tools/router-v2.tsv`

```bash
docker pull tenhkspark/gemma-4-v2:v2
cp gemma4-v2*.env gemma4-v2-serve.sh chat_template.jinja ~/gemma4-spark/
cd ~/gemma4-spark && ./gemma4-v2-serve.sh --env gemma4-v2-balanced.env up
./gemma4-v2-serve.sh smoke
```

The server listens on port 8890 (`/v1/chat/completions`). Router: `python3 tools/router.py --config tools/router-v2.tsv --listen 0.0.0.0:8899`; the balanced limit in `router-v2.tsv` is 32,768 tokens.

## Speculative decoding (MTP) setting

`GEMMA4_MTP=2` is the default and was the fastest or close to it on every workload we measured (table above).

`GEMMA4_MTP=8` (`gemma4-v2-balanced-mtp8.env`) suits structured extraction, template fill-in and log summaries: about 100 tok/s single-stream and 934 tok/s at concurrency 32 on 270 distinct prompts. On open-ended chat it is slower (48.9 against 62.9 tok/s single-stream), and on coding prompts it is 8-10% lower at concurrency 8 and 32. Choose per workload; it is one environment variable. Keep `GEMMA4_PREFIX_CACHE=0` in the balanced profiles.

## Getting concise replies

Gemma 4 is verbose by default. Four request-side knobs shorten replies:

- A short system prompt, for example `Answer in 3 sentences, no preamble.`
- `chat_template_kwargs: {"enable_thinking": false}` turns thinking off for the request.
- `max_tokens` caps the reply length.
- `reasoning_effort` and `thinking_token_budget` keep thinking short when it is on.

```bash
curl -s http://localhost:8890/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "gemma4",
  "messages": [
    {"role": "system", "content": "Answer in 2 sentences, no preamble."},
    {"role": "user", "content": "Why is the sky blue?"}],
  "max_tokens": 300,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

With that request a test answer was 47 tokens.

## Quality

Balanced with `GEMMA4_MTP=2`: the real-use comparison against v1 passed in sample mode, and the 125-question benchmark was at parity (82.4% against 82.4%). One greedy-mode comparison was within noise but below our stricter internal threshold. With `GEMMA4_MTP=8` both real-use modes passed and the benchmark was 83.2% against 84.0%.

## Contributors

tenhkspark. GLM-5.3 and GLM-5.3-Flash (Z.ai) assisted with analysis, test sets and documentation.

Licenses: see `LICENSE` and `NOTICE`; use of the model remains subject to the Gemma Terms of Use.
