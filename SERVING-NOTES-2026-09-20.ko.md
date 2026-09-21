[日本語](SERVING-NOTES-2026-09-20.ja.md) | [한국어](SERVING-NOTES-2026-09-20.ko.md) | [中文](SERVING-NOTES-2026-09-20.zh.md)

# Gemma 4 26B A4B on DGX Spark — 서빙 기록(2026-09-20)

> 수치는 모두 같은 날의 실측에서 비롯된다. 구 측정계의 값(prefix cache 유효 상태에서의 동일 프롬프트 재전송, 소수 사례의 최댓값을 대푯값으로 한 것)은 철회 완료이며, 본문에는 포함하지 않는다.

환경: DGX Spark 1대(GB10 / sm_121 / 통합 메모리 128GB), vLLM 0.28.0을 업스트림 Dockerfile에서 빌드한 이미지 `tenhkspark/vllm-gb10:v0.28.0-sm121`(`docker pull`로 받을 수 있다, 압축 9.4 GB / 전개 약 30 GB), TP=1. 측정은 모두 노드 위(localhost)에서 실행.

## 구성(A+γ8)

- 가중치: NVIDIA 공식 NVFP4 체크포인트(`nvidia/Gemma-4-26B-A4B-NVFP4`)에 `tie_word_embeddings`를 해제해 lm_head를 NVFP4화한 차분을 더한 것
- MTP 투기 디코딩: `num_speculative_tokens=8`(γ=8), draft는 `google/gemma-4-26B-A4B-it-assistant`
- `--language-model-only`(vision tower를 읽어들이지 않음)
- KV 캐시 fp8
- `--gpu-memory-utilization 0.5` / `--max-num-seqs 32` / `--max-num-batched-tokens 8192` / `--max-model-len 32768` / prefix caching 무효

## 기동

### 메모리 예약(처음에 결정한다)

DGX Spark는 GPU와 OS가 같은 128GB 통합 메모리를 나눠 쓴다. `--gpu-memory-utilization`은 「그 몇 할을 vLLM이 기동 시에 미리 가져가는가」의 지정으로, 실제 부하와 관계없이 확보된 채로 남는다.

기본은 **0.5**(`gemma4.env`의 `GEMMA4_GPU_UTIL=0.50`). 예약 내역은 가중치 17.08 GiB(기동 로그 실측) ＋ MTP 드래프트 약 0.8GB ＋ KV 캐시 ＋ 작업 영역. 0.5에서의 실측: 기동 로그에 `GPU KV cache size: 607,998 tokens`라고 나온다(32K 윈도우 환산으로 약 18.6개분). 기동 후의 `free -g`는 available 51〜52GB로, OS 측에 여유가 남는다.

util을 올릴수록 KV 풀은 늘지만, 그만큼 OS 측이 깎인다. 0.9로 기동했을 때는 OS 측 여유가 수 GB까지 떨어져 함께 돌고 있던 프로세스가 죽었다(실측).

- OS에 여유를 남기고 싶다 → 0.5(기본)
- 동시에 유지하는 개수를 늘리고 싶다 → 0.6 또는 0.7

바꾸는 법은 `gemma4.env`의 `GEMMA4_GPU_UTIL` 한 줄뿐이다. 변경 후에는 컨테이너를 다시 띄워야 한다(`./serve.sh down` → `up`, READY까지 약 4분).

| GPU_UTIL | KV 토큰 | 32K 환산의 동시 개수 | OS 측 여유 |
|---|---:|---:|---:|
| 0.5(기본) | 607,998 | 약 18.6개 | 51〜52 GB |
| 0.6 | 837,957 | 약 25.6개 | 40 GB |
| 0.7 | 1,011,401 | 약 30.9개 | 27 GB |
| 0.9 | — | — | 수 GB. 함께 돌던 프로세스가 죽었다 |

기동 로그에서 확인하는 5점(원문으로 갖춰진 후에 측정・이용한다):

1. `enable_prefix_caching=False`
2. `num_speculative_tokens=8`(γ=8의 MTP 투기)
3. `language_model_only=True`
4. `quantization=modelopt_fp4`와 `FlashInferCutlassNvFp4` 커널의 선택
5. lm_head가 양자화 대상으로 유지되어 있는 것

서버는 `--host 0.0.0.0`으로 뜬다(인증 없음). 신뢰할 수 있는 네트워크 안에서 쓰거나 127.0.0.1에 머물게 한다.

## 속도

실무형 프롬프트(일본어 가공 업무 문서 3유형: JSON 추출・재무 템플릿 빈칸 채우기・감시 로그 요약) 30문서 × 2반복, 온도 0, TTFT 포함의 헤드라인 속도(생성 토큰 수 ÷ 요청의 전체 소요 시간), 노드 위에서 실행.

| 단발(C=1) | 실무형・자연 EOS | 짧은 출력 조건(출력 32 tok) |
|---|---:|---:|
| 공식 NVFP4 그대로・투기 off | 28.8 tok/s | 28.0 tok/s |
| 공식 NVFP4 그대로・γ8 | 100.5 tok/s | 미측정 |
| A+γ8 | **109.7 tok/s** | 미측정 |
| 투기 off 대조 | 35.8 tok/s | 미측정 |

투기 디코딩에 의한 증속률: **3.06배**(109.7 / 35.8). 공식 체크포인트를
그대로 돌린 경우에도 28.8 → 100.5의 **3.49배**. lm_head를 NVFP4로
한 효과는 같은 γ8끼리 100.5 → 109.7의 **+9.2%**, 투기 없음끼리 
28.8 → 35.8의 **+24.3%**. 즉 이 구성의 속도는 대부분 투기 디코딩에서
비롯되고, lm_head 양자화는 그 위에 1할을 얹는 위치다.

병렬은 정상 부하(완료분을 즉시 보충해 동시 실행 수를 유지)로 각 병렬도 60초. 동일 조건의 2회 실행에서 일치한 값.

| 동시 실행 수 | 합계 tok/s |
|---:|---:|
| C=8 | 496.7 |
| C=16 | 822.0 |
| C=32 | 1,080.8 |

본문의 속도・품질 표는 모두 util **0.6**으로 기동한 서버의 측정값이다. 기본을 0.5로 해도 무방한 근거는 2가지 있다. 단발은 KV 풀량에 의존하지 않고, 0.5 기동에서도 **104.5 tok/s**를 실측하고 있다(0.6의 대푯값과 오차 내). 또 C=32의 윈도우가 필요로 하는 KV는 약 5만 토큰으로, 0.5로 확보되는 607,998 토큰에 비해 충분히 작다.

## 장문(동일 구성・기본값)

| 입력 길이 | C=1 (tok/s) | C=8 합계 (tok/s) | C=8 요청 p50 |
|---:|---:|---:|---:|
| 1,024 | 109.7 | 미측정 | — |
| 8K | 30.6 | 62.9 | 14.9 s |
| 28K | 10.4 | 12.0 | 89.0 s |

KV 풀은 837,957 토큰을 확보하고 있으며, 느려지는 이유는 KV 부족이 아니라 prefill 율속이다. 실무상으로는 장문을 분할해 짧게 던지는 편이 빠르다.

## 실무 태스크(90건)

일본어 실무 태스크 90건(SQL에서의 테이블명 추출 db・재무 템플릿 빈칸 채우기 finance・감시 로그의 요불요 판정 monitor, 각 30건)을 온도 0으로 흘려, 정확성(passed)과 처리량을 쟀다.

| 동시 실행 수 | passed | 건/분 | 요청 p95 |
|---:|---:|---:|---:|
| C=1 | 84/90 | 7.59 | 13.98 s |
| C=8 | 82/90 | 35.90 | 20.01 s |
| C=32 | 80/90 | 83.40 | 29.88 s |

유형별 내역: db가 가장 잘 틀리고, 실패는 거의 「불필요한 테이블을 나열한다」(table_extra). 이 경향은 **BF16(양자화 없음)에서도 똑같이 나온다**(C=1에서 BF16 20/30・본 레시피 24/30, BF16 측 처리량은 2.82건/분・p95 29.36 s)이므로, 양자화에서 비롯된 것이 아니다.

## 품질

JGLUE valid + JMMLU. 각 n=2,434(JCommonsenseQA는 전체 건수의 1,119). 동일 문제・동일 seed・greedy 쌍으로 BF16과 비교해 차이(pt)를 기재 [*].

| 지표 | n | Δ (pt) | 단측 95% 하한 (pt) |
|---|---:|---:|---:|
| JSQuAD EM | 2,434 | −0.66 | −1.17 |
| JSQuAD char-F1 | 2,434 | −0.17 | −0.41 |
| JNLI acc(1회차) | 2,434 | −1.23 | −1.93 |
| JNLI acc(2회차) | 2,434 | −1.48 | −2.14 |
| JCommonsenseQA acc | 1,119 | −0.36 | −1.07 |
| JMMLU acc | 2,434 | −0.70 | −1.60 |

판정: **5개 지표 모두 점추정으로 −2pt 이내. JNLI는 하한이 −2pt를 밑돌기 때문에 통계적 비열등성은 확인되지 않았다.**

[*] 구간 추정의 정본은 쌍 bootstrap(B=2,000・seed 고정・단측 95% 하한). Δ는 BF16과의 쌍 차이의 점추정.

### 독립 세트

본 평가는 JGLUE valid에 대한 것이므로, 누락을 의심해 구성 선택에 쓰지 않은 JGLUE train split 유래의 holdout 500문제/태스크로 재확인했다. JMMLU는 train/valid의 구별이 구조상 존재하지 않아 holdout을 만들 수 없고, 독립 확인은 **JSQuAD・JNLI・JCommonsenseQA의 3개 지표에 그친다**.

| 지표 | Δ (pt) |
|---|---:|
| JSQuAD EM | +0.20 |
| JSQuAD char-F1 | −0.18 |
| JNLI acc | −1.60 |
| JCommonsenseQA acc | −0.20 |

## 재현(3대・약 13분)

- 같은 절차로, 다른 DGX Spark 3대 모두에서 기동・smoke・벤치가 통과했다(현황 확인→배치→기동→smoke→bench→철수까지 약 13분)
- 위의 설정 5점이 3대 모두 기동 로그의 원문으로 일치
- 벤치(C=1・3회 평균): 99.3 / 105.6 / 102.7 tok/s. boot→READY는 241초
- 가중치의 manifest md5 `571932348835310ce77799f70a4e9814`가 3대에서 일치(13파일 / 19,240,726,248 B)
- 가중치는 base에 lm_head 차분을 더한 구조이므로, base가 이미 있는 노드로는 431MB 전송과 하드링크로 약 1분. 전량 복사는 불필요
- 주의: temperature 0에서도 대마다 생성문이 미묘하게 갈라졌다(가중치의 sha256은 일치). 「측정 방법」의 출력 개체 편차를 참조

## 측정 방법

확정값은 모두 다음 절차로 취득했다.

- **prefix caching 무효**: 기동 로그에서 `enable_prefix_caching=False`를 원문으로 확인하고 나서 측정. 유효한 채로 동일 계열을 재전송하면 prefill이 캐시에서 돌아와 실력보다 크게 보인다. 입력 2,048・출력 32 조건에서는 동일 프롬프트를 재전송한 2회차가 20배 이상으로 뛰었다(이 값은 실력이 아니므로 본문의 표에는 싣지 않았다)
- **노드 위 실행**: 벤치는 측정 대상 노드 위에서 `http://127.0.0.1:8890`(localhost)를 호출한다
- **실무형 프롬프트**: 각 문서 2반복・온도 0・자연 EOS. 대푯값은 3유형 중앙값의 평균(최댓값은 쓰지 않는다)
- **정상 부하**: 병렬은 완료분을 즉시 보충해 동시 실행 수를 유지하는 방식으로, 각 병렬도 60초
- **투기 off 대조**: 같은 가중치・같은 플래그로 투기 없는 암을 병행시켜, 증속률과 출력의 정확성을 대조한다
- **출력의 개체 편차**: temperature 0에서도 대마다 생성문이 미묘하게 갈라지는 일이 있다. 가중치의 sha256은 일치하는데 출력이 갈라지며, KV 풀량의 차이에 따른 수치 오차가 원인으로 보인다. 「동일 가중치이면 어느 대에서도 동일 응답」이라고는 한정되지 않는다

## 컨테이너 빌드

자체 Dockerfile이 아니라, vLLM 업스트림 리포지토리의 `docker/Dockerfile`을 그대로 빌드한 것이다.

```bash
DOCKER_BUILDKIT=1 docker build . \
    --tag tenhkspark/vllm-gb10:v0.28.0-sm121 \
    --build-arg BUILD_BASE_IMAGE=pytorch/manylinuxaarch64-builder:cuda13.0 \
    --build-arg torch_cuda_arch_list=12.0 \
    --build-arg max_jobs=8 --build-arg nvcc_threads=2
```

- `torch_cuda_arch_list`는 **12.0**(12.1이 아니다). GB10은 sm_121이지만 12.0으로 동작한다
- 검증: `--entrypoint python3`으로 기동해, `torch.cuda.get_device_capability()`가 `(12, 1)`을 반환하는 것
- 빌드 소요 시간은 실측 미기록. 가중치는 `huggingface-cli download tenhkspark/gemma-4-26B-A4B-NVFP4-lmhead`, 컨테이너는 `docker pull tenhkspark/vllm-gb10:v0.28.0-sm121`으로 받을 수 있다

## 크기

가중치 19.2GB ＋ 드래프트 0.8GB ＋ 컨테이너 이미지 약 30GB ＝ 약 50GB. 실행 시는 약 55GB.
GPU에 올라가는 양은 기동 로그의 `Model loading took`에서 실측 17.08 GiB.

## 다른 GPU에서 돌리는 경우(시산)

이 절은 모두 시산이며, 실측은 DGX Spark에서만 하고 있다.

- Spark는 통합 메모리로 OS와 공유하므로 util을 낮게(기본 0.5) 한다. 디스크리트 GPU는 VRAM을 OS와 공유하지 않으므로, 0.85〜0.9까지 올려도 좋다
- 예약 내역의 기준: 가중치 17.08 GiB(기동 로그 실측) ＋ MTP 드래프트 약 0.8GB ＋ KV 캐시
- 64GB GPU: `--max-model-len 32768`로 KV 약 35GB(32K 윈도우 × 약 20개)가 기준
- 32GB GPU: `--max-model-len 8192`・`--max-num-seqs 8`에서 시작한다
- Spark 이외의 GPU에서는 실측하지 않았다

## 왜 lm_head인가

decode가 1 스텝 34.775 ms 걸리고 있어서, torch profiler로 내역을 뽑아 보았다.

| 커널 | ms/step | 비율 |
|---|---:|---:|
| cuBLAS BF16 GEMV | 28.64 | 82.5% |
| MoE(라우팅+expert GEMM) | 4.75 | 13.6% |
| attention(Triton) | 0.643 | 1.85% |

상위 15 커널로 99.1%를 설명할 수 있다. 시간의 대부분은 BF16 그대로 남아 있는
선형층의 읽기에 쓰이고 있었다.

NVIDIA 공식의 NVFP4는 routed expert만 양자화하고 있으며, config.json의
`quantization_config.ignore`에는 93 엔트리(전체 30층의 `mlp*` / `router*` /
`self_attn*`와 `lm_head`, vision 계열)가 나열되어 있다. safetensors의 헤더를 집계하면,
매 토큰마다 읽는 BF16은 5.35 GB:

| 부위 | GB |
|---|---:|
| QKVO projection | 2.51 |
| lm_head(tied embedding) | 1.48 |
| shared-expert dense MLP | 1.34 |
| router·norm | 0.02 |

5.35 GB ÷ 28.64 ms = 실효 186.8 GB/s. GB10의 273 GB/s에 대해 68.4%로,
커널의 효율이 아니라 읽는 양이 발목을 잡고 있다.

이 중 lm_head는 `tie_word_embeddings`를 떼어 별도 텐서로 만들면,
단독으로 NVFP4로 할 수 있다. 결과는 같은 γ8에서 100.5 → **109.7 tok/s**(+9.2%),
투기 없음끼리 비교하면 28.8 → **35.8 tok/s**(+24.3%).

효과가 없었던 시도도 기록해 둔다.

- MoE 커널을 MARLIN으로 강제해도, 비투기 단발은 변하지 않았다
- NVFP4 체크포인트에 `--quantization fp8`을 겹쳐 지정해도, 기동 로그는
  `quantization=modelopt_fp4` 그대로이며 지정은 무시된다
- vision tower(0.59 GB)는 text-only의 decode에서는 읽히지 않는다.
  `--language-model-only`를 붙여도 속도는 변하지 않았다

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
일본어 생성도 무너지지 않는다.

### attention을 FP8로 한 버전(품질을 최우선으로 할 경우)

전층 4bit에서 JNLI가 다소 떨어졌기 때문에, attention(QKVO)만 FP8로 되돌리고,
공유 MLP・routed expert・lm_head를 NVFP4로 한 버전도 만들었다. 캘리브레이션 데이터는 같은
일본어 365건.

| 지표 | 본 레시피 | attention FP8 버전 |
|---|---:|---:|
| 단발 tok/s | 109.7 | **117.5(+7.1%)** |
| 병렬 C=32 합계 tok/s | 1,080.8 | 약 절반 |
| JNLI의 페어 차(valid) | −1.23 / −1.48 pt | **−0.74 pt** |
| JNLI의 페어 차(독립 세트) | −1.60 pt | **−1.20 pt** |
| 5개 지표의 점추정 | 모두 −2pt 이내 | 모두 −2pt 이내 |

**단발과 품질 모두 본 레시피를 웃돌지만, 병렬이 약 절반이므로 채택하지 않았다.** 배치 용도에서는
합계 처리량이 중요하기 때문이다. 단발만 사용하고 품질을 최우선으로 하는 경우에는 이쪽이 더 나을
가능성이 있다. 가중치는 공개하지 않았다(만드는 방법은 본문의 절차와 동일하며, attention만
FP8 target으로 한다).

attention을 높은 정밀도로 되돌리면 품질이 돌아오고 단발이 약간 떨어진다는 트레이드오프는
전층 4bit 버전과의 비교에서도 나타나 있다(JNLI −1.64 → −0.74 pt, 단발 122.3 → 117.5).

## Tools used

Tools used — Claude Fable 5.1, GPT-6 (Astra), GLM-5.3-Flash. All code in this repository was written for this project.

## 사사・라이선스

- Google(Gemma 4 본체와 MTP 드래프트), NVIDIA(NVFP4 체크포인트와 DGX Spark), vLLM(서빙 엔진)에 감사한다
- Gemma 4의 기본 라이선스는 Apache License 2.0
- 본문의 수치는 모두 이 리포지토리 안의 실측에서 비롯된다
