# Gemma 4 26B NVFP4 × MTP (A+γ8) — DGX Spark BRING-UP

対象構成（A+γ8）: NVIDIA 公式 NVFP4 チェックポイントに、tie を外した lm_head を NVFP4 化して足したもの。
γ=8（MTP）、`--language-model-only`、KV fp8、`--gpu-memory-utilization 0.5`、`--max-num-seqs 32`、
`--max-num-batched-tokens 8192`、`--max-model-len 32768`、prefix caching 無効。

## 1. 必要なもの

- DGX Spark 1 台（GB10 / 統合メモリ 128GB）
- ディスク約 50GB（重み 19.2GB + ドラフト 0.8GB + コンテナイメージ約 30GB）
- 実行時メモリ 55GB

## 2. コンテナ

自前 Dockerfile ではなく、vLLM 上流リポジトリの `docker/Dockerfile` をビルドしたものを使う。
vLLM ソースツリー直下で:

```bash
DOCKER_BUILDKIT=1 docker build . \
  --file docker/Dockerfile \
  --target vllm-openai \
  --platform linux/arm64 \
  --tag wabi/vllm-gb10:v0.28.0-sm121 \
  --build-arg BUILD_BASE_IMAGE=pytorch/manylinuxaarch64-builder:cuda13.0 \
  --build-arg torch_cuda_arch_list=12.0 \
  --build-arg max_jobs=8 \
  --build-arg nvcc_threads=2 \
  --progress=plain
```

- 要点: `torch_cuda_arch_list` は **12.0**（12.1 ではない）。GB10 は sm_121 だが 12.0 で動く。
- ビルド所要時間は実測未記録。
- 配布形式は公開時に確定。

検証（ENTRYPOINT が `vllm serve` なので `--entrypoint` で上書きする）:

```bash
docker run --rm --runtime nvidia --gpus all \
  --entrypoint python3 wabi/vllm-gb10:v0.28.0-sm121 -c \
  "import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.get_device_capability())"
```

期待: `0.28.0 <torch 版> (12, 1)` — `torch.cuda.get_device_capability()` が `(12, 1)` を返すこと。

## 3. 重みの取得

- 配布形式は公開時に確定。
- 構造は「base（NVIDIA 公式 NVFP4）に lm_head を足した差分」。MTP ドラフト（assistant モデル、約 0.8GB）は別ディレクトリ。
- base のチェックポイントが既にあるノードへは、差分 431MB の転送と shard 2 本を base から hardlink で再構成して約 1 分。全量コピーは不要。
- 正本の manifest: 13 ファイル / 19,240,726,248 B / md5 `571932348835310ce77799f70a4e9814`。

## 4. 起動

### メモリ予約（先に読む・最初に決める）

DGX Spark は GPU と OS が同じ 128GB の統合メモリを分け合う。`--gpu-memory-utilization` は「その何割を vLLM が起動時に先取りするか」の指定で、実際の負荷に関係なく、確保されたままになる。

既定は **0.5**（`gemma4.env` の `GEMMA4_GPU_UTIL=0.50`）。予約の内訳は重み 17.08 GiB（起動ログ実測）＋ MTP ドラフト 約 0.8GB ＋ KV キャッシュ ＋ 作業領域。0.5 での実測: 起動ログに `GPU KV cache size: 607,998 tokens`（32K 窓換算で約 18.6 本分）、起動後 `free -g` の available は 51〜52GB。

util を上げるほど KV プールは増えるが、その分 OS 側は削られる。0.9 で起動した時は OS 側の空きが数 GB まで落ち、同居していたプロセスが落ちた（実測）。

- OS に余裕を残したい → 0.5（既定）
- 同時に保持する本数を増やしたい → 0.6 か 0.7

変え方は `gemma4.env` の `GEMMA4_GPU_UTIL` の 1 行だけ。変更後はコンテナの立て直しが必要（`./serve.sh down` → `up`、READY まで約 4 分）。

| GPU_UTIL | KV トークン | 32K 換算の同時本数 | OS 側の空き |
|---|---:|---:|---:|
| 0.5（既定） | 607,998 | 約 18.6 本 | 51〜52 GB |
| 0.6 | 837,957 | 約 25.6 本 | 40 GB |
| 0.7 | 1,011,401 | 約 30.9 本 | 27 GB |
| 0.9 | — | — | 数 GB。同居プロセスが落ちた |

起動は 1 コマンド（ノード上で実行）:

```bash
cd gemma4-spark   # このディレクトリ（重みのパスは gemma4.env の GEMMA4_MODEL / GEMMA4_MTP_DIR）
./serve.sh up
```

`serve.sh` の `--gpu-memory-utilization` 既定は 0.50（`gemma4.env` の `GEMMA4_GPU_UTIL`）。READY まで自動で待つ（実測 241 秒）。停止は `./serve.sh down`。

## 5. 動作確認

smoke（日本語 1 問を投げて tok/s を表示）:

```bash
./serve.sh smoke
```

bench 1 セル（例: 入力約 1K・C=1）:

```bash
python3 bench-cell.py
```

起動ログで確認すべき 5 点（`docker logs gemma4-serve`）:

1. `'enable_prefix_caching': False`
2. `'num_speculative_tokens': 8`（`SpeculativeConfig(method='mtp', …)`）
3. `'language_model_only': True`
4. `quantization=modelopt_fp4` と `Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM`
5. `Gemma4 MTP: keeping draft model's own lm_head (draft_dim != backbone_dim).` — lm_head を量子化対象に保持した重みが読めていること

## 6. 実測値

以下の速度・質の表はすべて util **0.6** で起動したサーバの測定値。既定を 0.5 にしても差し支えない根拠は 2 つある。単発は KV プール量に依存せず、0.5 の起動でも 104.5 tok/s を実測している（0.6 の代表値と誤差内）。また C=32 の窓が要る KV は約 5 万トークンで、0.5 で確保される 607,998 トークンに対して十分に小さい。

### 速度（実務型 30 文書 × 2 反復、TTFT 込みの見出し速度、ノード上実行）

| 条件 | 値 |
|---|---|
| 単発 代表値 | 109.7 tok/s |
| 単発 投機 off 対照 | 35.8 tok/s（増速率 2.91 倍） |
| 並列 C=8（定常負荷 60 秒、2 回一致） | 496.7 tok/s |
| 並列 C=16 | 822.0 tok/s |
| 並列 C=32 | 1,080.8 tok/s |

短出力枠（出力 32 tok）は別欄の測定系で扱う。

### 長文（同一構成・既定値）

| 入力長 | C=1 (tok/s) | C=8 合計 (tok/s) |
|---|---|---|
| 約 1K | 109.7 | — |
| 8K | 30.6 | 62.9（p50 14.9s） |
| 28K | 10.4 | 12.0（p50 89.0s） |

KV は 837,957 トークン確保済みで、律速は KV 不足ではなく prefill。実務上は長文を分割して短く投げる方が速い。

### 質（JGLUE valid + JMMLU、BF16 とのペア差）

各 n=2,434（JCommonsenseQA は全 1,119）:

| 指標 | Δ (pt) | 片側 95% 下限 |
|---|---|---|
| JSQuAD EM | −0.66 | −1.17 |
| JSQuAD F1 | −0.17 | −0.41 |
| JNLI（1 回目） | −1.23 | −1.93 |
| JNLI（2 回目） | −1.48 | −2.14 |
| JCommonsenseQA | −0.36 | −1.07 |
| JMMLU | −0.70 | −1.60 |

判定: **5 指標とも点推定で −2pt 以内。JNLI は下限が −2pt をまたぐため、統計的非劣性は確認できていない。**

独立セット: 構成の選択に使っていない JGLUE train split 由来の holdout 500 問/タスクで再確認。JMMLU は train/valid の区別が構造上存在しないため holdout を作れず、独立確認は JSQuAD・JNLI・JCommonsenseQA の 3 指標どまり。

| 指標 | Δ (pt) |
|---|---:|
| JSQuAD EM | +0.20 |
| JSQuAD char-F1 | −0.18 |
| JNLI acc | −1.60 |
| JCommonsenseQA acc | −0.20 |

## 7. 再現の実測（3 台・13 分）

2026-09-20、3 台で「取得 → 起動 → smoke → bench → 撤収」まで約 13 分（15:38Z–15:51Z）。

- bench mean of 3（C=1）: **99.3 / 105.6 / 102.7 tok/s**、boot→READY **241 秒**（3 台とも）。
- manifest md5 `571932348835310ce77799f70a4e9814` が 3 台で一致（13 ファイル / 19,240,726,248 B）。
- §5 の設定 5 点が 3 台の起動ログ原文で一致。
- 重みは差分構造のため、base のあるノードへは 431MB 転送 + hardlink で約 1 分。全量コピーは不要。

## 8. 注意

- **temperature 0 でも台ごとに生成文が微妙に分岐する**（重みの sha256 は一致）。KV cache 量の台ごとの差による数値誤差と見られる。**同一出力は保証されない。**

## 9. 他の GPU で動かす場合（試算）

この節はすべて試算であり、実測は DGX Spark のみで行っている。

- Spark は統合メモリで OS と共有するため util を低め（既定 0.5）にする。ディスクリート GPU は VRAM を OS と共有しないため、0.85〜0.9 まで上げてよい
- 予約の内訳の目安: 重み 17.08 GiB（起動ログ実測）＋ MTP ドラフト 約 0.8GB ＋ KV キャッシュ
- 64GB の GPU: `--max-model-len 32768` で KV 約 35GB（32K 窓 × 約 20 本）が目安
- 32GB の GPU: `--max-model-len 8192`・`--max-num-seqs 8` から始める
- Spark 以外の GPU では実測していない

## 謝辞

Google・NVIDIA・vLLM の 3 者の成果物の上に成り立っています。

Tools used — Claude Fable 5.1, GPT-6 (Astra), GLM-5.3-Flash. All code in this repository was written for this project.
