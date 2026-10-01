# Gemma 4 26B-A4B on DGX Spark, v2

适用于单台 DGX Spark上 Gemma 4 26B-A4B NVFP4 版本的部署配置。模型权重与 v1 相同。v1：git 标签 `v1`。

## 从 v1 升级到 v2

| tok/s（除非另有说明） | v1 | v2 |
|---|---|---|
| 上下文长度 | 32,768 | 262,144 |
| 首 token 时间，32k 提示词 | 12.4 s | 6.1 s（prefill-first 配置） |
| 首 token 时间，128k 提示词 | 不支持 | 56 s（prefill-first 配置） |
| 首 token 时间，250k 提示词 | 不支持 | 182 s（prefill-first 配置） |
| 解码，日语聊天，tok/s，并发数 1 / 32 | 45.9 / 616 | 62.9 / 850 |
| 解码，编程 | 67.8 / 644 | 69.5 / 725 |
| 解码，工具调用 | 54.6 / 193 | 61.4 / 272 |
| 解码，长文档 | 29.1 / 152 | 38.0 / 448 |
| 解码，结构化提取（`GEMMA4_MTP=8` 选项） | 109.7 / 1,080.8（按发布数据） | 约 100 / 934 |
| 质量（基准测试和实际使用检查） | 基线 | 持平 |
| 冷启动 | 约 4 min | 约 4 min |

在均衡配置下，2k / 8k / 30k 提示词的首 token 时间与 v1 相同（0.32 s / 1.57 s / 12.3 s）。稳定性：通过路由器并发处理 32 个短请求和 4 个长请求（最长 131k），持续 15 分钟，0 个错误，0 次超时。

## 升级时需要更改的内容

- 镜像：[tenhkspark/gemma-4-v2:v2](https://hub.docker.com/r/tenhkspark/gemma-4-v2)
- 环境文件：`gemma4-v2.env` 加一个配置文件，`gemma4-v2-balanced.env`（默认）或 `gemma4-v2-prefill-first.env`（长提示词）
- 服务脚本：`gemma4-v2-serve.sh`，聊天模板 `chat_template.jinja`
- 路由器（可选，多节点）：`tools/router.py` 和 `tools/router-v2.tsv`

```bash
docker pull tenhkspark/gemma-4-v2:v2
cp gemma4-v2*.env gemma4-v2-serve.sh chat_template.jinja ~/gemma4-spark/
cd ~/gemma4-spark && ./gemma4-v2-serve.sh --env gemma4-v2-balanced.env up
./gemma4-v2-serve.sh smoke
```

服务器监听端口 8890（`/v1/chat/completions`）。路由器：`python3 tools/router.py --config tools/router-v2.tsv --listen 0.0.0.0:8899`；`router-v2.tsv` 中的均衡配置上限为 32,768 个 token。

## 投机解码（MTP）设置

`GEMMA4_MTP=2` 是默认值，在我们测量的所有工作负载中，它的速度都是最快或接近最快（见上表）。

`GEMMA4_MTP=8`（`gemma4-v2-balanced-mtp8.env`）适用于结构化提取、模板填充和日志摘要：在 270 个不同提示词上，单路约 100 tok/s，并发数为 32 时为 934 tok/s。在开放式聊天中速度较慢（单路 48.9 对 62.9 tok/s）；在编程提示词上，并发数为 8 和 32 时低 8-10%。按工作负载选择；只需设置一个环境变量。均衡配置中保持 `GEMMA4_PREFIX_CACHE=0`。

## 让回复更简洁

Gemma 4 默认会生成较长的回复。可以通过四个请求侧参数缩短回复：

- 简短的系统提示词，例如 `Answer in 3 sentences, no preamble.`
- `chat_template_kwargs: {"enable_thinking": false}` 会关闭本次请求的思考功能。
- `max_tokens` 限制回复长度。
- 启用思考时，使用 `reasoning_effort` 和 `thinking_token_budget` 保持思考简短。

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

使用该请求时，测试回复为 47 个 token。

## 质量

在 `GEMMA4_MTP=2` 的均衡配置下，与 v1 的实际使用对比在抽样模式下通过，125 个问题的基准测试结果持平（82.4% 对 82.4%）。一次贪心模式对比的差异在噪声范围内，但低于我们更严格的内部阈值。使用 `GEMMA4_MTP=8` 时，两种实际使用模式均通过，基准测试结果为 83.2% 对 84.0%。

## 贡献者

tenhkspark。GLM-5.3 和 GLM-5.3-Flash（Z.ai）协助了分析、测试集和文档编写。

许可证：参见 `LICENSE` 和 `NOTICE`；模型的使用仍受 Gemma 使用条款约束。
