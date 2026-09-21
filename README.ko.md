[English](README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [中文](README.zh.md)

# gemma4-spark — Gemma 4 26B A4B NVFP4(lm_head 분리판)를 NVIDIA DGX Spark에서

## 배포하는 것

**A+γ8** 이라는 구성입니다.

- NVIDIA 공식 NVFP4 체크포인트(`nvidia/Gemma-4-26B-A4B-NVFP4`)를 기반으로 한다
- `tie_word_embeddings`를 해제하고 **lm_head도 NVFP4로 양자화**한 것
  (appendix의 `untie-lmhead-fp8.py`가 이 가중치의 만드는 방법)
- MTP 투기 디코딩, `num_speculative_tokens = 8`(γ8), 드래프트는
  `google/gemma-4-26B-A4B-it-assistant`
- `--language-model-only`(vision tower를 읽지 않는다)
- KV 캐시는 FP8

## 수록물

- `serve.sh` / `gemma4.env` / `gemma4.small.env` — 단일 노드 서빙용
  하네스(`up` / `down` / `status` / `smoke`)
- `BRING-UP.md` — 브링업 절차 전문(일본어)
- `SERVING-NOTES-2026-09-20.{ja,ko,zh}.md` — 서빙 기록. ko/zh는
  확정판에서 번역
- `MODEL-CARD.md` — 모델 카드
- `untie-lmhead-fp8.py` — appendix: 이 가중치의 만드는 방법
- `bench-cell.py` — 브링업 확인용 C=1 벤치 1셀
- `LICENSE`, `NOTICE`, `LICENSE-CHECK.md`, `check.sh`, `upload.sh`

## 요건

- DGX Spark 1대(GB10・통합 메모리 128 GB), 또는 Blackwell 세대의 GPU
  (NVFP4 연산에는 그 세대가 필요)
- 디스크 약 50 GB(가중치 19.2 GB + 드래프트 0.8 GB + 컨테이너 약 30 GB)
- 컨테이너 이미지 — vLLM 상류의 `docker/Dockerfile`을 빌드
  (커맨드와 인수는 `BRING-UP.md` §2)

## 퀵스타트

1. **가중치를 받는다**. 배포 형식은 공개 시에 확정. 검증용 manifest는
   13 파일 / 19,240,726,248 B / md5 `571932348835310ce77799f70a4e9814`.
2. **컨테이너를 준비**(`BRING-UP.md` §2).
3. **`./serve.sh up`** — 32 GB급 디스크리트 GPU라면
   `./serve.sh up --env gemma4.small.env`.

동작 확인: `./serve.sh smoke`(일본어 1문을 던져 tok/s를 표시).

## 실측치(DGX Spark, 2026-09-20)

### 속도

실무형 프롬프트 3형 × 30문서 × 2반복, 온도 0, TTFT 포함 헤드라인 속도:

| 단발(C=1) | tok/s |
|---|---:|
| A+γ8(이 레시피) | **109.7** |
| 같은 구성・투기 없음(대조) | 35.8 |

투기 디코딩에 의한 증속률: **2.91 배**. 정상 부하(각 병렬도 60초)에서의 합계:

| 동시 실행 수 | 합계 tok/s |
|---:|---:|
| C=8 | 496.7 |
| C=16 | 822.0 |
| C=32 | 1,080.8 |

### 실무 태스크(일본어 90건, 온도 0)

| 동시 실행 수 | passed | 건/분 |
|---:|---:|---:|
| C=1 | 84/90 | 7.59 |
| C=8 | 82/90 | 35.90 |
| C=32 | 80/90 | 83.40 |

### 장문

| 입력 길이 | C=1 (tok/s) | C=8 합계 (tok/s) |
|---:|---:|---:|
| 8K | 30.6 | 62.9 |
| 28K | 10.4 | 12.0 |

KV 부족이 아니라 prefill 율속 — 장문은 나눠서 보낸다.

### 품질(BF16과의 페어 차, pt)

JGLUE valid + JMMLU. 각 n = 2,434(JCommonsenseQA는 전 1,119).

| 지표 | Δ (pt) | 단측 95% 하한 |
|---|---:|---:|
| JSQuAD EM | −0.66 | −1.17 |
| JSQuAD char-F1 | −0.17 | −0.41 |
| JNLI acc(1회차) | −1.23 | −1.93 |
| JNLI acc(2회차) | −1.48 | −2.14 |
| JCommonsenseQA acc | −0.36 | −1.07 |
| JMMLU acc | −0.70 | −1.60 |

**5개 지표 모두 점추정으로 −2 pt 이내. JNLI는 하한이 −2 pt를 밑돌기
때문에 통계적 비열등성은 확인되어 있지 않다.**

독립 세트 — 구성 선택에 사용하지 않은 holdout 태스크당 500문항:

| 지표 | Δ (pt) |
|---|---:|
| JSQuAD EM | +0.20 |
| JSQuAD char-F1 | −0.18 |
| JNLI acc | −1.60 |
| JCommonsenseQA acc | −0.20 |

JMMLU는 대상 외: 구조상 train/valid 구별이 없어 holdout을 만들 수 없다.

### 메모리 예약

DGX Spark는 통합 메모리로, GPU와 OS가 같은 128 GB를 나눠 쓴다.
`--gpu-memory-utilization`은 기동 시 vLLM이 선점하는 비율로,
부하와 무관하게 확보된 채로 남는다. 기본값은 **0.5**(`gemma4.env`의
`GEMMA4_GPU_UTIL`): 기동 로그에 `GPU KV cache size: 607,998 tokens`라고
나오고 OS 측 여유는 51-52 GB가 남는다. util을 올리면 KV 풀은 늘지만
OS 측이 깎인다 — 0.9에서는 OS 여유가 수 GB까지 떨어져 같은 노드의
프로세스가 죽었다(실측).

| GPU_UTIL | KV 토큰 | 32K 환산 | OS 측 여유 |
|---|---:|---:|---:|
| 0.5(기본) | 607,998 | 약 18.6본 | 51-52 GB |
| 0.6 | 837,957 | 약 25.6본 | 40 GB |
| 0.7 | 1,011,401 | 약 30.9본 | 27 GB |
| 0.9 | — | — | 수 GB. 같은 노드의 프로세스가 죽었다 |

위의 속도・품질 표는 util 0.6으로 측정한 값. 기본값 0.5와 모순되지
않는 근거는 2개: 단발 속도는 KV 풀 크기에 의존하지 않아 0.5에서도
104.5 tok/s를 실측(오차 범위). C=32의 창이 필요로 하는 KV는 약 5만
토큰으로 0.5의 607,998에 대해 충분.

### 크기

배포물은 약 19 GB. GPU에 올라가는 양은 기동 로그 실측으로 17.08 GiB
(`Model loading took`).

## 다른 GPU에서 돌리는 경우(시산)

이 절은 전부 시산이며, 실측은 DGX Spark만. NVFP4 연산에는
Blackwell 세대의 GPU가 필요하다.

- 디스크리트 GPU는 VRAM을 OS와 공유하지 않으므로 util은
  0.85〜0.9까지 올려도 된다
- 내역의 기준: 가중치 17.08 GiB(기동 로그 실측) + MTP 드래프트 약 0.8 GB + KV 풀
- 64 GB GPU: `--max-model-len 32768`로 KV 약 35 GB(32K 창
  약 20본)이 기준
- 32 GB GPU: `--max-model-len 8192`・`--max-num-seqs 8`에서 시작
  — `gemma4.small.env`가 그대로 그 값을 가진다

`serve.sh`를 쓰지 않는 경우, 조립되는 커맨드는 다음과 같다
(gemma4.small.env 설정. `/checkpoint`・`/checkpoint-mtp`는
마운트된 가중치와 드래프트의 경로):

```bash
vllm serve /checkpoint --served-model-name gemma4 --host 0.0.0.0 --port 8890 --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 8192 --enable-chunked-prefill --no-enable-prefix-caching --language-model-only --trust-remote-code --reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice --limit-mm-per-prompt '{"image":0,"audio":0}' --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 --speculative-config '{"method":"mtp","model":"/checkpoint-mtp","num_speculative_tokens":8}'
```

## 제한・주의

- 장문은 prefill 율속으로 떨어진다. 나눠서 보낼 것(위의 표).
- 온도 0에서도 기체마다 생성문이 미묘하게 갈라진다(가중치는 동일.
  갈라지는 쪽은 서빙 측).
- 실무 태스크 평가에서 db(SQL에서 테이블명 추출)는 불필요한 테이블을
  나열하는 경향이 있다. BF16(양자화 없음)에서도 같은 경향이 나온다
  (C=1로 BF16 20/30・이 레시피 24/30)므로 양자화 유래가 아니다.
- 이 API에는 인증이 없다. 서버는 모든 인터페이스에서 응답하므로,
  신뢰할 수 있는 네트워크에서 돌리거나 localhost에 둘 것.

## 다음 과제

### 전 계층 4bit ＋ 일본어 캘리브레이션(시도했지만 채택하지 않음)

본 레시피의 가중치는 attention과 공유 MLP를 BF16 그대로 남겨 두었다. 그 부분까지
포함해 모두 NVFP4로 바꾸고, 일본어 캘리브레이션 데이터 365건과 오차 보상이 붙은 양자화(GPTQ 계열)로
다시 만든 판을 실측했다.

| 지표 | 본 레시피 | 전 계층 4bit 판 |
|---|---:|---:|
| 단발 tok/s | 109.7 | **122.3(+11.5%)** |
| 병렬 C=32 합계 tok/s | 1,080.8 | **586.5(약 절반)** |
| JNLI의 페어 차 | −1.23 / −1.48 pt | −1.64 pt |
| 5개 지표의 점추정 | 모두 −2pt 이내 | 모두 −2pt 이내 |

**품질은 유지했지만 병렬에서 크게 떨어졌기 때문에 채택하지 않았다.** 단발에서는 빠르고,
일본어 생성도 무너지지 않는다. 병렬만 느린 이유는 밝혀내지 못했다(단발은 대역 율속,
병렬은 계산 율속에 들어가기 때문이라는 견해는 있지만 미검증).

같은 측정 방식으로 병렬의 하락을 설명하고 피할 수 있다면, 단발의 +11.5%를 유지한 채
병렬도 가져갈 수 있을 가능성이 있다.

## 라이선스

Apache License 2.0 — `LICENSE`와 `NOTICE`를 참조. 모델 본체의
이용에는 Gemma Terms of Use와 금지 용도 정책이 계속 적용된다.

## Tools used

Tools used — Claude Fable 5.1, GPT-6 (Astra), GLM-5.3-Flash. All code in this repository was written for this project.

## 사사

Google(Gemma 4 본체와 MTP 드래프트), NVIDIA(NVFP4 체크포인트와
DGX Spark), vLLM(서빙 엔진)에 감사한다.
