[日本語](SERVING-NOTES-2026-09-20.ja.md) | [한국어](SERVING-NOTES-2026-09-20.ko.md) | [中文](SERVING-NOTES-2026-09-20.zh.md)

# Gemma 4 26B A4B on DGX Spark — 服务化记录（2026-09-20）

> 所有数值均来自当日实测。旧测量体系的数值（启用 prefix cache 后重发同一提示词、以少量样本的最大值充当代表值的做法）已撤回，本文不再收录。

环境：1 台 DGX Spark（GB10 / sm_121 / 统一内存 128GB），vLLM 0.28.0 为使用上游 Dockerfile 构建的镜像 `tenhkspark/vllm-gb10:v0.28.0-sm121`（可用 `docker pull` 获取，压缩 9.4 GB / 展开约 30 GB），TP=1。所有测量均在节点上（localhost）执行。

## 配置（A+γ8）

- 权重：在 NVIDIA 官方 NVFP4 检查点（`nvidia/Gemma-4-26B-A4B-NVFP4`）之上，叠加了移除 `tie_word_embeddings` 并将 lm_head 进行 NVFP4 量化的差分
- MTP 投机解码：`num_speculative_tokens=8`（γ=8），draft 为 `google/gemma-4-26B-A4B-it-assistant`
- `--language-model-only`（不加载 vision tower）
- KV 缓存 fp8
- `--gpu-memory-utilization 0.5` / `--max-num-seqs 32` / `--max-num-batched-tokens 8192` / `--max-model-len 32768` / prefix caching 禁用

## 启动

### 内存预留（最先决定）

DGX Spark 的 GPU 与 OS 共享同一块 128GB 统一内存。`--gpu-memory-utilization` 指定的是"其中多大比例由 vLLM 在启动时预先占用"，与实际负载无关，一经占用便不再释放。

默认值为 **0.5**（`gemma4.env` 中的 `GEMMA4_GPU_UTIL=0.50`）。预留构成为：权重 17.08 GiB（启动日志实测）＋ MTP 草稿约 0.8GB ＋ KV 缓存 ＋ 工作区。0.5 时的实测：启动日志输出 `GPU KV cache size: 607,998 tokens`（按 32K 窗口换算约 18.6 个并发）。启动后 `free -g` 显示 available 51〜52GB，OS 侧仍有余量。

util 调得越高 KV 池越大，但 OS 侧会相应被压缩。以 0.9 启动时 OS 侧空闲降至数 GB，同机驻留的进程被杀掉（实测）。

- 想给 OS 留余量 → 0.5（默认）
- 想增加同时保留的并发数 → 0.6 或 0.7

修改方法只改 `gemma4.env` 中 `GEMMA4_GPU_UTIL` 这一行。修改后需要重建容器（`./serve.sh down` → `up`，到 READY 约 4 分钟）。

| GPU_UTIL | KV token 数 | 按 32K 换算的并发数 | OS 侧空闲 |
|---|---:|---:|---:|
| 0.5（默认） | 607,998 | 约 18.6 个 | 51〜52 GB |
| 0.6 | 837,957 | 约 25.6 个 | 40 GB |
| 0.7 | 1,011,401 | 约 30.9 个 | 27 GB |
| 0.9 | — | — | 数 GB，同机驻留进程被杀 |

启动日志中需确认的 5 点（与原文核对一致后再测量、使用）：

1. `enable_prefix_caching=False`
2. `num_speculative_tokens=8`（γ=8 的 MTP 投机）
3. `language_model_only=True`
4. `quantization=modelopt_fp4` 与 `FlashInferCutlassNvFp4` 内核被选中
5. lm_head 仍在量化对象中

服务器以 `--host 0.0.0.0` 启动（无认证）。仅在可信网络内使用，或限定在 127.0.0.1。

## 速度

实务型提示词（日语虚构业务文书 3 种：JSON 抽取、财务模板填空、监控日志摘要）30 份文档 × 2 次重复、温度 0、含 TTFT 的标称速度（生成 token 数 ÷ 请求全程耗时），节点上执行。

| 单发（C=1） | 实务型、自然 EOS | 短输出档（输出 32 tok） |
|---|---:|---:|
| 官方 NVFP4 原样・投机 off | 28.8 tok/s | 28.0 tok/s |
| 官方 NVFP4 原样・γ8 | 100.5 tok/s | 未测 |
| A+γ8 | **109.7 tok/s** | 未测 |
| 投机 off 对照 | 35.8 tok/s | 未测 |

投机解码带来的加速比：**3.06 倍**（109.7 / 35.8）。即便直接运行官方
检查点，也有 28.8 → 100.5 的 **3.49 倍**。把 lm_head 改为 NVFP4 的
效果，在同为 γ8 时是 100.5 → 109.7 的 **+9.2%**，在同为关闭投机时是
28.8 → 35.8 的 **+24.3%**。也就是说这套配置的速度大部分来自投机解码，
lm_head 的量化是在其之上再加一成。

并发采用稳态负载（完成即补发以维持并发数），每个并发度 60 秒。数值为同条件两次运行一致的结果。

| 并发数 | 合计 tok/s |
|---:|---:|
| C=8 | 496.7 |
| C=16 | 822.0 |
| C=32 | 1,080.8 |

本文速度与质量表中的数值均来自以 util **0.6** 启动的服务器。默认设为 0.5 亦无妨，依据有二：单发不依赖 KV 池容量，以 0.5 启动也实测到 **104.5 tok/s**（与 0.6 的代表值在误差范围内）；且 C=32 窗口所需 KV 约 5 万 token，相对 0.5 预留的 607,998 token 足够小。

## 长文本（同一配置、默认值）

| 输入长度 | C=1 (tok/s) | C=8 合计 (tok/s) | C=8 请求 p50 |
|---:|---:|---:|---:|
| 1,024 | 109.7 | 未测 | — |
| 8K | 30.6 | 62.9 | 14.9 s |
| 28K | 10.4 | 12.0 | 89.0 s |

KV 池已预留 837,957 token，速度下降的原因并非 KV 不足，而是 prefill 瓶颈。实务上将长文本切分后以短输入提交更快。

## 实务任务（90 件）

以温度 0 运行了 90 件日语实务任务（从 SQL 抽取表名 db、财务模板填空 finance、监控日志必要性判定 monitor，各 30 件），测量了正确性（passed）与吞吐量。

| 并发数 | passed | 件/分 | 请求 p95 |
|---:|---:|---:|---:|
| C=1 | 84/90 | 7.59 | 13.98 s |
| C=8 | 82/90 | 35.90 | 20.01 s |
| C=32 | 80/90 | 83.40 | 29.88 s |

按类型细分：**finance（从表格提取数值）在所有条件下均为 30/30**；monitor 在 C=1 时为 30/30（始终回答“需要处理”的基线为 15/30）。db 最容易失败，失败几乎全是“多列出了表”（table_extra）。该倾向在 **BF16（无量化）下同样出现**（C=1 时 BF16 20/30、本方案 24/30，BF16 侧吞吐为 2.82 件/分、p95 29.36 s），因此并非量化所致。

## 质量

JGLUE valid + JMMLU。各 n=2,434（JCommonsenseQA 为全量 1,119）。以同一题目、同一 seed、greedy 的配对与 BF16 比较，记录差值（pt）[*]。

| 指标 | n | Δ (pt) | 单侧 95% 下限 (pt) |
|---|---:|---:|---:|
| JSQuAD EM | 2,434 | −0.66 | −1.17 |
| JSQuAD char-F1 | 2,434 | −0.17 | −0.41 |
| JNLI acc（第 1 次） | 2,434 | −1.23 | −1.93 |
| JNLI acc（第 2 次） | 2,434 | −1.48 | −2.14 |
| JCommonsenseQA acc | 1,119 | −0.36 | −1.07 |
| JMMLU acc | 2,434 | −0.70 | −1.60 |

判定：**5 项指标的点估计均在 −2pt 以内。但 JNLI 的下限跨越 −2pt，统计非劣性尚未得到确认。**

[*] 区间估计的正本为配对 bootstrap（B=2,000、seed 固定、单侧 95% 下限）。Δ 为与 BF16 配对差的点估计。

### 独立集

本评估针对的是 JGLUE valid，为排查遗漏，使用未参与配置选择的 JGLUE train split 派生的 holdout 500 题/任务重新确认。JMMLU 在结构上不存在 train/valid 之分，无法构造 holdout，因此独立确认**仅覆盖 JSQuAD、JNLI、JCommonsenseQA 这 3 项指标**。

| 指标 | Δ (pt) |
|---|---:|
| JSQuAD EM | +0.20 |
| JSQuAD char-F1 | −0.18 |
| JNLI acc | −1.60 |
| JCommonsenseQA acc | −0.20 |

## 复现（3 台、约 13 分钟）

- 按相同步骤，另外 3 台 DGX Spark 全部通过启动、smoke、bench（现状确认→部署→启动→smoke→bench→撤收共约 13 分钟）
- 上述 5 项配置在 3 台的启动日志原文中全部一致
- bench（C=1、3 次平均）：99.3 / 105.6 / 102.7 tok/s。boot→READY 为 241 秒
- 权重 manifest md5 `571932348835310ce77799f70a4e9814` 在 3 台一致（13 个文件 / 19,240,726,248 B）
- 权重为 base 加 lm_head 差分的结构，已有 base 的节点只需传输 431MB 并做硬链接，约 1 分钟。无需全量复制
- 注意：即使 temperature 0，各机器的生成文本仍有细微分叉（权重 sha256 一致）。参见"测量方法"的输出个体差异

## 测量方法

所有定稿数值均按以下步骤取得。

- **prefix caching 禁用**：在启动日志中原文确认 `enable_prefix_caching=False` 后再测量。启用状态下重发同一序列会使 prefill 命中缓存返回，速度看起来高于实际能力。在输入 2,048、输出 32 的条件下，重发同一提示词的第 2 次跃升至 20 倍以上（该数值并非实际能力，故未收入本文的表格）
- **节点上执行**：bench 从被测节点上请求 `http://127.0.0.1:8890`（localhost）
- **实务型提示词**：每份文档 2 次重复、温度 0、自然 EOS。代表值取 3 种类型中位数的平均（不使用最大值）
- **稳态负载**：并发采用完成即补发以维持并发数的方式，每个并发度 60 秒
- **投机 off 对照**：用相同权重、相同 flags 并跑一个无投机臂，对照加速比与输出正确性
- **输出个体差异**：即使 temperature 0，各机器的生成文本也可能有细微分叉。权重 sha256 一致但输出不同，据信是 KV 池容量差异引起的数值误差所致。"同一权重则任意机器同一响应"并不保证成立

## 容器构建

并非自写 Dockerfile，而是直接构建 vLLM 上游仓库的 `docker/Dockerfile`。

```bash
DOCKER_BUILDKIT=1 docker build . \
    --tag tenhkspark/vllm-gb10:v0.28.0-sm121 \
    --build-arg BUILD_BASE_IMAGE=pytorch/manylinuxaarch64-builder:cuda13.0 \
    --build-arg torch_cuda_arch_list=12.0 \
    --build-arg max_jobs=8 --build-arg nvcc_threads=2
```

- `torch_cuda_arch_list` 为 **12.0**（不是 12.1）。GB10 虽是 sm_121，但 12.0 即可运行
- 验证：以 `--entrypoint python3` 启动，确认 `torch.cuda.get_device_capability()` 返回 `(12, 1)`
- 构建耗时未实测记录。权重可用 `huggingface-cli download tenhkspark/gemma-4-26B-A4B-NVFP4-lmhead`、容器可用 `docker pull tenhkspark/vllm-gb10:v0.28.0-sm121` 获取

## 体积

权重 19.2GB ＋ 草稿 0.8GB ＋ 容器镜像约 30GB ＝ 约 50GB。运行时约 55GB。
载入 GPU 的量以启动日志 `Model loading took` 实测为 17.08 GiB。

## 在其他 GPU 上运行时（估算）

本节内容均为估算，实测仅在 DGX Spark 上进行。

- Spark 以统一内存与 OS 共享，因此 util 设低（默认 0.5）。独立 GPU 的 VRAM 不与 OS 共享，可提高至 0.85〜0.9
- 预留构成参考：权重 17.08 GiB（启动日志实测）＋ MTP 草稿约 0.8GB ＋ KV 缓存
- 64GB GPU：`--max-model-len 32768` 时 KV 约 35GB（32K 窗口 × 约 20 个并发）为参考值
- 32GB GPU：从 `--max-model-len 8192`、`--max-num-seqs 8` 起步
- 未在 Spark 以外的 GPU 上实测

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

### 全层 4bit ＋ 日语校准（试过，未采用）

本配方的权重将 attention 与共享 MLP 保留为 BF16。我们也实测了把这部分
一并转为 NVFP4、并用 365 条日语校准数据与带误差补偿的量化（GPTQ 系）
重新制作的版本。

| 指标 | 本配方 | 全层 4bit 版 |
|---|---:|---:|
| 单发 tok/s | 109.7 | **122.3（+11.5%）** |
| 并行 C=32 合计 tok/s | 1,080.8 | **586.5（约一半）** |
| JNLI 的配对差 | −1.23 / −1.48 pt | −1.64 pt |
| 5 项指标的点估计 | 均在 −2pt 以内 | 均在 −2pt 以内 |

**质量得以保持，但并行下降很大，因此未予采用。** 单发更快，日语生成也不会
崩坏。

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

## Tools used

Tools used — Claude Fable 5.1, GPT-6 (Astra), GLM-5.3-Flash. All code in this repository was written for this project.

## 致谢、许可

- 感谢 Google（Gemma 4 本体与 MTP 草稿）、NVIDIA（NVFP4 检查点与 DGX Spark）、vLLM（服务引擎）
- Gemma 4 的基础许可证为 Apache License 2.0
- 本文数值均来自本仓库内的实测
