# Gemma 4 26B-A4B on DGX Spark, v2

단일 DGX Spark에서 Gemma 4 26B-A4B의 NVFP4 빌드를 제공하기 위한 설정입니다. 모델 가중치는 v1과 동일합니다. v1은 git tag `v1`로 이용할 수 있습니다.

## v1에서 v2로

| tok/s 별도 표기가 없는 경우 | v1 | v2 |
|---|---|---|
| 컨텍스트 길이 | 32,768 | 262,144 |
| 첫 토큰까지 시간, 32k 프롬프트 | 12.4 s | 6.1 s (prefill 우선 프로파일) |
| 첫 토큰까지 시간, 128k 프롬프트 | 지원 안 함 | 56 s (prefill 우선 프로파일) |
| 첫 토큰까지 시간, 250k 프롬프트 | 지원 안 함 | 182 s (prefill 우선 프로파일) |
| 디코드, 일본어 채팅, 동시성 1 / 32에서 tok/s | 45.9 / 616 | 62.9 / 850 |
| 디코드, 코딩 | 67.8 / 644 | 69.5 / 725 |
| 디코드, 도구 호출 | 54.6 / 193 | 61.4 / 272 |
| 디코드, 긴 문서 | 29.1 / 152 | 38.0 / 448 |
| 디코드, 구조화된 추출 (`GEMMA4_MTP=8` 옵션) | 109.7 / 1,080.8 (게시된 값 기준) | 약 100 / 934 |
| 품질 (벤치마크 및 실제 사용 검사) | 기준선 | 동등 |
| 콜드 스타트 | 약 4 min | 약 4 min |

균형 프로파일에서 2k / 8k / 30k 프롬프트의 첫 토큰까지 시간은 v1과 같습니다 (0.32 s / 1.57 s / 12.3 s). 안정성: 라우터를 통해 짧은 요청 32개와 긴 요청 4개(최대 131k)를 동시에 15분간 처리했으며, 오류 0건, 시간 초과 0건이었습니다.

## 업그레이드할 때 변경할 항목

- 이미지: [tenhkspark/gemma-4-v2:v2](https://hub.docker.com/r/tenhkspark/gemma-4-v2)
- Env 파일: `gemma4-v2.env`와 프로파일 하나, `gemma4-v2-balanced.env` (기본값) 또는 `gemma4-v2-prefill-first.env` (긴 프롬프트)
- 제공 스크립트: `gemma4-v2-serve.sh`, 채팅 템플릿 `chat_template.jinja`
- 라우터 (선택 사항, 여러 노드): `tools/router.py`와 `tools/router-v2.tsv`

```bash
docker pull tenhkspark/gemma-4-v2:v2
cp gemma4-v2*.env gemma4-v2-serve.sh chat_template.jinja ~/gemma4-spark/
cd ~/gemma4-spark && ./gemma4-v2-serve.sh --env gemma4-v2-balanced.env up
./gemma4-v2-serve.sh smoke
```

서버는 포트 8890 (`/v1/chat/completions`)에서 요청을 수신합니다. 라우터: `python3 tools/router.py --config tools/router-v2.tsv --listen 0.0.0.0:8899`; `router-v2.tsv`의 균형 제한은 32,768 토큰입니다.

## 투기적 디코딩 (MTP) 설정

`GEMMA4_MTP=2`가 기본값이며, 측정한 모든 작업 부하에서 가장 빠르거나 그에 가까웠습니다 (위 표 참조).

`GEMMA4_MTP=8` (`gemma4-v2-balanced-mtp8.env`)은 구조화된 추출, 템플릿 채우기, 로그 요약에 적합합니다. 서로 다른 프롬프트 270개에서 단일 스트림은 약 100 tok/s, 동시성 32에서는 934 tok/s였습니다. 자유 형식 채팅에서는 더 느리고 (단일 스트림 48.9 대 62.9 tok/s), 코딩 프롬프트에서는 동시성 8과 32에서 8-10% 낮습니다. 작업 부하별로 선택하세요. 환경 변수 하나로 설정할 수 있습니다. 균형 프로파일에서는 `GEMMA4_PREFIX_CACHE=0`을 유지하세요.

## 간결한 답변 얻기

Gemma 4는 기본적으로 답변이 깁니다. 요청 측 설정 네 가지로 답변을 짧게 할 수 있습니다.

- 짧은 시스템 프롬프트. 예: `Answer in 3 sentences, no preamble.`
- `chat_template_kwargs: {"enable_thinking": false}`를 설정하면 해당 요청에서 사고 기능이 꺼집니다.
- `max_tokens`는 답변 길이를 제한합니다.
- 사고 기능이 켜져 있을 때는 `reasoning_effort`와 `thinking_token_budget`으로 사고 길이를 짧게 유지할 수 있습니다.

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

이 요청으로 테스트한 답변은 47 tokens였습니다.

## 품질

`GEMMA4_MTP=2` 균형 설정에서 v1과의 실제 사용 비교는 샘플 모드를 통과했고, 125개 질문 벤치마크는 동등한 결과였습니다 (82.4% 대 82.4%). 탐욕 모드 비교 한 건은 노이즈 범위 안이었지만 더 엄격한 내부 기준에는 미치지 못했습니다. `GEMMA4_MTP=8`에서는 실제 사용 두 모드 모두 통과했고 벤치마크는 83.2% 대 84.0%였습니다.

## 기여자

tenhkspark. GLM-5.3과 GLM-5.3-Flash(Z.ai)가 분석, 테스트 세트, 문서 작성을 도왔습니다.

라이선스: `LICENSE`와 `NOTICE`를 참조하세요. 모델 사용에는 Gemma 이용 약관이 계속 적용됩니다.
