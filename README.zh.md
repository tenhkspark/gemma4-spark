[English](README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [中文](README.zh.md)

单台 DGX Spark 运行 Gemma 4 26B A4B。
官方 NVFP4 原样: 单发 28.8 tok/s。
本配方: 单发 109.7 tok/s,32 并发合计 1,080.8 tok/s。

# gemma4-spark — 在 NVIDIA DGX Spark 上运行 Gemma 4 26B A4B NVFP4(lm_head 分离版)

## 发布内容

我们称之为 **A+γ8** 的配置:

- 以 NVIDIA 官方 NVFP4 检查点(`nvidia/Gemma-4-26B-A4B-NVFP4`)为基础
- 解除 `tie_word_embeddings`,**lm_head 也量化为 NVFP4**
  (附录 `untie-lmhead-fp8.py` 说明该权重的制作方法)
- MTP 投机解码,`num_speculative_tokens = 8`(γ8),草稿模型为
  `google/gemma-4-26B-A4B-it-assistant`
- `--language-model-only`(不加载 vision tower)
- KV 缓存为 FP8

## 收录文件

- `serve.sh` / `gemma4.env` / `gemma4.small.env` — 单节点服务
  启动器(`up` / `down` / `status` / `smoke`)
- `BRING-UP.md` — 完整 bring-up 步骤(日语)
- `SERVING-NOTES-2026-09-20.{ja,ko,zh}.md` — 服务记录。ko/zh 译自
  定稿版
- `MODEL-CARD.md` — 模型卡
- `untie-lmhead-fp8.py` — 附录: 该权重的制作方法
- `bench-cell.py` — bring-up 验证用的 C=1 bench 单元
- `LICENSE`、`NOTICE`、`LICENSE-CHECK.md`、`check.sh`、`upload.sh`

## 要求

- 一台 DGX Spark(GB10・统一内存 128 GB),或 Blackwell 世代 GPU
  (NVFP4 运算需要该世代)
- 磁盘约 50 GB(权重 19.2 GB + 草稿 0.8 GB + 容器约 30 GB)
- 容器镜像 — `docker pull tenhkspark/vllm-gb10:v0.28.0-sm121`
  (压缩后 9.4 GB)。若自行构建,命令与参数见 `BRING-UP.md` §2

## 快速开始

1. **获取权重**(`huggingface-cli download tenhkspark/gemma-4-26B-A4B-NVFP4-lmhead --local-dir ./gemma4-lmhead`)。校验用 manifest 为
   13 个文件 / 19,240,726,248 B / md5 `571932348835310ce77799f70a4e9814`。
2. **获取容器**(`docker pull tenhkspark/vllm-gb10:v0.28.0-sm121`)。
   自行构建的步骤见 `BRING-UP.md` §2。
3. **`./serve.sh up`** — 若为 32 GB 级独立显卡则用
   `./serve.sh up --env gemma4.small.env`。

冒烟检查: `./serve.sh smoke`(发送一道日语问题并显示 tok/s)。

## 实测值(DGX Spark,2026-09-20)

### 速度

实用型提示词 3 种 × 30 文档 × 2 次重复,温度 0,含 TTFT 的标称速度:

| 单发(C=1) | tok/s |
|---|---:|
| 官方 NVFP4 原样・关闭投机 | 28.8 |
| 官方 NVFP4 原样・γ8 | 100.5 |
| A+γ8(本配方) | **109.7** |
| 同配置・关闭投机(对照) | 35.8 |

投机解码增速比: **3.06 倍**(109.7 / 35.8)。官方原样也有
28.8 → 100.5 的 **3.49 倍**。把 lm_head 改为 NVFP4 的效果是
100.5 → 109.7 的 **+9.2%**(同为关闭投机时 28.8 → 35.8 的 +24.3%)。稳定负载(各并发度 60 秒)下的合计:

| 并发数 | 合计 tok/s |
|---:|---:|
| C=8 | 496.7 |
| C=16 | 822.0 |
| C=32 | 1,080.8 |

### 实用任务(90 个日语任务,温度 0)

| 并发数 | passed | 件/分 |
|---:|---:|---:|
| C=1 | 84/90 | 7.59 |
| C=8 | 82/90 | 35.90 |
| C=32 | 80/90 | 83.40 |

### 长文

| 输入长度 | C=1 (tok/s) | C=8 合计 (tok/s) |
|---:|---:|---:|
| 8K | 30.6 | 62.9 |
| 28K | 10.4 | 12.0 |

瓶颈是 prefill 而非 KV 不足 — 长文请拆分输入。

### 质量(与 BF16 的成对差,pt)

JGLUE valid + JMMLU。各 n = 2,434(JCommonsenseQA 为全部 1,119)。

| 指标 | Δ (pt) | 单侧 95% 下限 |
|---|---:|---:|
| JSQuAD EM | −0.66 | −1.17 |
| JSQuAD char-F1 | −0.17 | −0.41 |
| JNLI acc(第 1 次) | −1.23 | −1.93 |
| JNLI acc(第 2 次) | −1.48 | −2.14 |
| JCommonsenseQA acc | −0.36 | −1.07 |
| JMMLU acc | −0.70 | −1.60 |

**5 项指标的点估计均在 −2 pt 以内。JNLI 下限跨越 −2 pt,
因此统计非劣性未确认。**

独立集 — 未用于配置选择的 holdout,每任务 500 题:

| 指标 | Δ (pt) |
|---|---:|
| JSQuAD EM | +0.20 |
| JSQuAD char-F1 | −0.18 |
| JNLI acc | −1.60 |
| JCommonsenseQA acc | −0.20 |

JMMLU 不在范围内: 其结构没有 train/valid 划分,无法构建 holdout。

### 内存预留

DGX Spark 是统一内存: GPU 与 OS 共用同一个 128 GB。
`--gpu-memory-utilization` 是 vLLM 启动时预留的比例,无论负载如何
都会保持占用。默认值为 **0.5**(`gemma4.env` 中的
`GEMMA4_GPU_UTIL`): 启动日志报告 `GPU KV cache size: 607,998
tokens`,OS 侧剩余 51-52 GB。提高 util 会增大 KV 池但压缩 OS 侧
— 0.9 时 OS 侧只剩数 GB,同机进程被终止(实测)。

| GPU_UTIL | KV token | 32K 换算 | OS 侧余量 |
|---|---:|---:|---:|
| 0.5(默认) | 607,998 | 约 18.6 个 | 51-52 GB |
| 0.6 | 837,957 | 约 25.6 个 | 40 GB |
| 0.7 | 1,011,401 | 约 30.9 个 | 27 GB |
| 0.9 | — | — | 数 GB;同机进程被终止 |

上面的速度与质量表是在 util 0.6 下测得的。默认 0.5 与之不矛盾的
依据有两条: 单发速度不依赖 KV 池大小,0.5 下实测 104.5 tok/s
(误差范围内);C=32 的窗口所需 KV 约 5 万 token,相对 0.5 下的
607,998 足够。

### 大小

分发物约 19 GB。GPU 上占用量为启动日志实测的 17.08 GiB
(`Model loading took`)。

## 在其他 GPU 上运行(估算)

本节全部为估算,实测仅在 DGX Spark 上进行。NVFP4 运算需要
Blackwell 世代 GPU。

- 独立显卡不与 OS 共享显存,util 可提高到 0.85〜0.9
- 预算构成: 权重 17.08 GiB(启动日志实测) + MTP 草稿约 0.8 GB + KV 池
- 64 GB GPU: `--max-model-len 32768` 时 KV 约 35 GB(约 20 个
  32K 窗口)为基准
- 32 GB GPU: 从 `--max-model-len 8192`・`--max-num-seqs 8` 起步
  — `gemma4.small.env` 携带的正是这些值

不使用 `serve.sh` 时,组装出的命令如下(gemma4.small.env 配置。
`/checkpoint`・`/checkpoint-mtp` 为挂载的权重与草稿路径):

```bash
vllm serve /checkpoint --served-model-name gemma4 --host 0.0.0.0 --port 8890 --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 8192 --enable-chunked-prefill --no-enable-prefix-caching --language-model-only --trust-remote-code --reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice --limit-mm-per-prompt '{"image":0,"audio":0}' --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 --speculative-config '{"method":"mtp","model":"/checkpoint-mtp","num_speculative_tokens":8}'
```

## 限制与注意

- 长文受 prefill 律速影响而变慢。请拆分输入(上表)。
- 即使温度 0,不同机器的生成文本也会略有分叉(权重相同,
  差异出在服务端)。
- 实用任务评估中,db(从 SQL 提取表名)有列出多余表名的倾向。
  BF16(无量化)也有相同倾向(C=1 下 BF16 20/30・本配方 24/30),
  因此并非量化所致。
- 该 API 没有认证。服务器监听所有接口,请在可信网络中运行或
  绑定 localhost。

## 为什么是 lm_head

decode 每一步耗时 34.775 ms，因此用 torch profiler 取了内部分解。

| 内核 | ms/step | 占比 |
|---|---:|---:|
| cuBLAS BF16 GEMV | 28.64 | 82.5% |
| MoE（路由＋expert GEMM） | 4.75 | 13.6% |
| attention（Triton） | 0.643 | 1.85% |

前 15 个内核即可解释 99.1% 的时间。大部分时间仍留在 BF16 的
线性层读取上。

NVIDIA 官方的 NVFP4 只量化 routed expert，config.json 的
`quantization_config.ignore` 中列有 93 个条目（全部 30 层的 `mlp*` / `router*` /
`self_attn*`、`lm_head` 以及 vision 系）。统计 safetensors 的头部后，每个 token 要读取的 BF16 为 5.35 GB：

| 部位 | GB |
|---|---:|
| QKVO 投影 | 2.51 |
| lm_head（tied embedding） | 1.48 |
| shared-expert dense MLP | 1.34 |
| router、norm | 0.02 |

5.35 GB ÷ 28.64 ms = 实效 186.8 GB/s。相对 GB10 的 273 GB/s 为 68.4%，
起作用的不是内核的效率，而是读取量。

其中 lm_head 只要去掉 `tie_word_embeddings`、改为独立张量，
就可以单独转换成 NVFP4。结果是同样在 γ8 下从 100.5 提升到 **109.7 tok/s**（+9.2%），
无投机的两者对比下从 28.8 提升到 **35.8 tok/s**（+24.3%）。

也把没有奏效的手段记录下来。

- 即使把 MoE 内核强制为 MARLIN，非投机的单发也没有变化
- 即使在 NVFP4 检查点上叠加 `--quantization fp8`，启动日志仍为
  `quantization=modelopt_fp4`，指定被忽略
- vision tower（0.59 GB）在 text-only 的 decode 中不会被读取。
  即使加上 `--language-model-only`，速度也没有变化

## 下一步课题

### 全层 4bit ＋ 日语校准(试过,未采用)

本配方的权重将 attention 与共享 MLP 保留为 BF16。我们也实测了把这部分一并
转为 NVFP4、并用 365 条日语校准数据与带误差补偿的量化(GPTQ 系)重新制作的版本。

| 指标 | 本配方 | 全层 4bit 版 |
|---|---:|---:|
| 单发 tok/s | 109.7 | **122.3(+11.5%)** |
| 并行 C=32 合计 tok/s | 1,080.8 | **586.5(约一半)** |
| JNLI 的配对差 | −1.23 / −1.48 pt | −1.64 pt |
| 5 项指标的点估计 | 均在 −2pt 以内 | 均在 −2pt 以内 |

**质量得以保持,但并行下降很大,因此未予采用。** 单发更快,日语生成也不会崩坏。

### 将 attention 改为 FP8 的版本（优先质量时）

由于全层 4bit 下 JNLI 略有下降，我们还制作了仅将 attention（QKVO）改回 FP8、
共享 MLP・routed expert・lm_head 改为 NVFP4 的版本。校准数据同样是
365 条日语。

| 指标 | 本配方 | attention FP8 版 |
|---|---:|---:|
| 单发 tok/s | 109.7 | **117.5（+7.1%）** |
| 并行 C=32 合计 tok/s | 1,080.8 | 约一半 |
| JNLI 的配对差（valid） | −1.23 / −1.48 pt | **−0.74 pt** |
| JNLI 的配对差（独立集） | −1.60 pt | **−1.20 pt** |
| 5 项指标的点估计 | 全部在 −2pt 以内 | 全部在 −2pt 以内 |

**该版本在单发和质量上都超过本配方，但由于并行只有约一半，因此没有采用。** 因为在
批量用途中，总吞吐量才是关键。如果只使用单发且优先质量，则该版本可能更好。
权重未公开（制作方法与正文步骤相同，只需将 attention 设为
FP8 target）。

将 attention 恢复到较高精度可以恢复质量但单发略有下降，这一权衡在
与全层 4bit 版的比较中也有体现（JNLI −1.64 → −0.74 pt、单发 122.3 → 117.5）。

## 许可证

Apache License 2.0 — 见 `LICENSE` 与 `NOTICE`。模型本体的使用
仍受 Gemma Terms of Use 及禁止用途政策约束。

## Tools used

Tools used — Claude Fable 5.1, GPT-6 (Astra), GLM-5.3-Flash. All code in this repository was written for this project.

## 致谢

感谢 Google(Gemma 4 本体与 MTP 草稿)、NVIDIA(NVFP4 检查点与
DGX Spark)、vLLM(服务引擎)。
