[English](README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [中文](README.zh.md)

DGX Spark 1 台で Gemma 4 26B A4B。
公式 NVFP4 そのまま: 単発 28.8 tok/s。
本レシピ: 単発 109.7 tok/s、32 並列で合計 1,080.8 tok/s。

# gemma4-spark — Gemma 4 26B A4B NVFP4(lm_head 分離版)を NVIDIA DGX Spark で

## 何を配るか

**A+γ8** という構成です。

- NVIDIA 公式 NVFP4 チェックポイント(`nvidia/Gemma-4-26B-A4B-NVFP4`)を土台にする
- `tie_word_embeddings` を外し、**lm_head も NVFP4 に量子化**したもの
  (appendix の `untie-lmhead-fp8.py` がこの重みの作り方)
- MTP 投機デコード、`num_speculative_tokens = 8`(γ8)、ドラフトは
  `google/gemma-4-26B-A4B-it-assistant`
- `--language-model-only`(vision tower を読み込まない)
- KV キャッシュは FP8

## 収録物

- `serve.sh` / `gemma4.env` / `gemma4.small.env` — 1 台サービング用の
  ハーネス(`up` / `down` / `status` / `smoke`)
- `BRING-UP.md` — ブリングアップ手順の全文(日本語のみ)
- `SERVING-NOTES-2026-09-20.{ja,ko,zh}.md` — サービング記録。ko/zh は
  確定版からの翻訳
- `MODEL-CARD.md` — モデルカード
- `untie-lmhead-fp8.py` — appendix: この重みの作り方
- `bench-cell.py` — ブリングアップ確認用の C=1 ベンチ 1 セル
- `LICENSE`、`NOTICE`、`LICENSE-CHECK.md`、`check.sh`、`upload.sh`

## 要件

- DGX Spark 1 台(GB10・統合メモリ 128 GB)、または Blackwell 世代の GPU
  (NVFP4 の演算はその世代が必要)
- ディスク約 50 GB(重み 19.2 GB + ドラフト 0.8 GB + コンテナ約 30 GB)
- コンテナイメージ — `docker pull tenhkspark/vllm-gb10:v0.28.0-sm121`
  (圧縮 9.4 GB)。自分でビルドする場合のコマンドと引数は `BRING-UP.md` §2

## クイックスタート

1. **重みを取得**（`huggingface-cli download tenhkspark/gemma-4-26B-A4B-NVFP4-lmhead --local-dir ./gemma4-lmhead`）。検証用 manifest は
   13 ファイル / 19,240,726,248 B / md5 `571932348835310ce77799f70a4e9814`。
2. **コンテナを取得**（`docker pull tenhkspark/vllm-gb10:v0.28.0-sm121`）。
   自前ビルドの手順は `BRING-UP.md` §2。
3. **`./serve.sh up`** — 32 GB 級のディスクリート GPU なら
   `./serve.sh up --env gemma4.small.env`。

動作確認: `./serve.sh smoke`(日本語 1 問を投げて tok/s を表示)。

## 実測値(DGX Spark、2026-09-20)

### 速度

実務型プロンプト 3 型 × 30 文書 × 2 反復、温度 0、TTFT 込みの見出し速度:

| 単発(C=1) | tok/s |
|---|---:|
| 公式 NVFP4 そのまま・投機なし | 28.8 |
| 公式 NVFP4 そのまま・γ8 | 100.5 |
| A+γ8(本レシピ) | **109.7** |
| 同構成・投機なし(対照) | 35.8 |

投機デコードによる増速率: **3.06 倍**(109.7 / 35.8)。公式そのままでも
28.8 → 100.5 の **3.49 倍**。lm_head を NVFP4 にした効果は
100.5 → 109.7 の **+9.2%**(投機なし同士なら 28.8 → 35.8 の +24.3%)。定常負荷(各並列度 60 秒)での合計:

| 同時実行数 | 合計 tok/s |
|---:|---:|
| C=8 | 496.7 |
| C=16 | 822.0 |
| C=32 | 1,080.8 |

### 実務タスク(日本語 90 件、温度 0)

| 同時実行数 | passed | 件/分 |
|---:|---:|---:|
| C=1 | 84/90 | 7.59 |
| C=8 | 82/90 | 35.90 |
| C=32 | 80/90 | 83.40 |

カテゴリ別では、finance（表からの数値抽出）は全条件で 30/30、monitor は
C=1 で 30/30（「要対応」と答え続ける場合は 15/30）。db は最も失敗が多く、
ほとんどが余計なテーブル名を挙げる失敗だった。この傾向は BF16 でも同じ
（C=1 で BF16 20/30、本レシピ 24/30）なので、量子化由来ではない。

測定時は **prefix caching を無効**にし、起動ログで
`enable_prefix_caching=False` を確認すること。有効のまま同じプロンプトを
再送すると prefill がキャッシュから返り、2 回目以降の速度が実力より大きく
見える。実測では入力 2,048・出力 32 の同一プロンプト再送で 2 回目が 20 倍以上
になった。上表にはその値を含めていない。詳細は
`SERVING-NOTES-2026-09-20.ja.md` を参照。

### 長文

| 入力長 | C=1 (tok/s) | C=8 合計 (tok/s) |
|---:|---:|---:|
| 8K | 30.6 | 62.9 |
| 28K | 10.4 | 12.0 |

KV 不足ではなく prefill 律速 — 長文は分割して投げる。

### 質(BF16 とのペア差、pt)

JGLUE valid + JMMLU。各 n = 2,434(JCommonsenseQA は全 1,119)。

| 指標 | Δ (pt) | 片側 95% 下限 |
|---|---:|---:|
| JSQuAD EM | −0.66 | −1.17 |
| JSQuAD char-F1 | −0.17 | −0.41 |
| JNLI acc(1 回目) | −1.23 | −1.93 |
| JNLI acc(2 回目) | −1.48 | −2.14 |
| JCommonsenseQA acc | −0.36 | −1.07 |
| JMMLU acc | −0.70 | −1.60 |

**5 指標とも点推定で −2 pt 以内。JNLI は下限が −2 pt をまたぐため、
統計的非劣性は確認できていない。**

独立セット — 構成の選択に使っていない holdout 500 問/タスク:

| 指標 | Δ (pt) |
|---|---:|
| JSQuAD EM | +0.20 |
| JSQuAD char-F1 | −0.18 |
| JNLI acc | −1.60 |
| JCommonsenseQA acc | −0.20 |

JMMLU は対象外: 構造上 train/valid の区別がなく、holdout を作れない。

### メモリ予約

DGX Spark は統合メモリで、GPU と OS が同じ 128 GB を分け合う。
`--gpu-memory-utilization` は起動時に vLLM が先取りする割合で、
負荷に関係なく確保されたままになる。既定は **0.5**(`gemma4.env` の
`GEMMA4_GPU_UTIL`): 起動ログに `GPU KV cache size: 607,998 tokens`
と出て、OS 側の空きは 51-52 GB 残る。util を上げると KV プールは
増えるが OS 側が削られる — 0.9 では OS の空きが数 GB まで落ちて
同居プロセスが落ちた(実測)。

| GPU_UTIL | KV トークン | 32K 換算 | OS 側の空き |
|---|---:|---:|---:|
| 0.5(既定) | 607,998 | 約 18.6 本 | 51-52 GB |
| 0.6 | 837,957 | 約 25.6 本 | 40 GB |
| 0.7 | 1,011,401 | 約 30.9 本 | 27 GB |
| 0.9 | — | — | 数 GB。同居プロセスが落ちた |

上の速度・質の表は util 0.6 で測定した値。既定 0.5 と矛盾しない
根拠は 2 つ: 単発速度は KV プール量に依存せず 0.5 でも 104.5 tok/s
を実測(誤差内)。C=32 の窓が要る KV は約 5 万トークンで 0.5 の
607,998 に対して十分。

### サイズ

配布物は約 19 GB。GPU に載る量は起動ログ実測で 17.08 GiB
(`Model loading took`)。

## 他の GPU で動かす場合(試算)

この節はすべて試算であり、実測は DGX Spark のみ。NVFP4 の演算には
Blackwell 世代の GPU が要る。

- ディスクリート GPU は VRAM を OS と共有しないので util は
  0.85〜0.9 まで上げてよい
- 内訳の目安: 重み 17.08 GiB（起動ログ実測）+ MTP ドラフト 約 0.8 GB + KV プール
- 64 GB の GPU: `--max-model-len 32768` で KV 約 35 GB(32K 窓
  約 20 本)が目安
- 32 GB の GPU: `--max-model-len 8192`・`--max-num-seqs 8` から
  — `gemma4.small.env` がそのままその値を持つ

`serve.sh` を使わない場合、組み立てられるコマンドは
(gemma4.small.env 設定。`/checkpoint`・`/checkpoint-mtp` は
マウントされた重みとドラフトのパス):

```bash
vllm serve /checkpoint --served-model-name gemma4 --host 0.0.0.0 --port 8890 --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 8192 --enable-chunked-prefill --no-enable-prefix-caching --language-model-only --trust-remote-code --reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice --limit-mm-per-prompt '{"image":0,"audio":0}' --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 --speculative-config '{"method":"mtp","model":"/checkpoint-mtp","num_speculative_tokens":8}'
```

## 制限・注意

- 長文は prefill 律速で落ちる。分割して投げること(上の表)。
- temperature 0 でも機体ごとに生成文が微妙に分岐する(重みは同一。
  差が出るのはサービング側)。
- 実務タスク評価で、db(SQL からのテーブル名抽出)は余計なテーブルを
  挙げる傾向がある。BF16(量子化なし)でも同じ傾向が出る(C=1 で
  BF16 20/30・本レシピ 24/30)ので、量子化由来ではない。
- この API には認証がない。サーバは全インターフェースで応答するので、
  信頼できるネットワークで動かすか localhost に留める。

## なぜ lm_head なのか

decode が 1 ステップ 34.775 ms かかっていたので、torch profiler で内訳を取った。

| カーネル | ms/step | 割合 |
|---|---:|---:|
| cuBLAS BF16 GEMV | 28.64 | 82.5% |
| MoE（ルーティング＋expert GEMM） | 4.75 | 13.6% |
| attention（Triton） | 0.643 | 1.85% |

上位 15 カーネルで 99.1% を説明できる。時間の大半は BF16 のまま残っている
線形層の読み出しに使われていた。

NVIDIA 公式の NVFP4 は routed expert だけを量子化しており、config.json の
`quantization_config.ignore` に 93 エントリ（全 30 層の `mlp*` / `router*` /
`self_attn*` と `lm_head`、vision 系）が並ぶ。safetensors のヘッダを集計すると、
毎トークン読む BF16 は 5.35 GB:

| 部位 | GB |
|---|---:|
| QKVO 射影 | 2.51 |
| lm_head（tied embedding） | 1.48 |
| shared-expert dense MLP | 1.34 |
| router・norm | 0.02 |

5.35 GB ÷ 28.64 ms = 実効 186.8 GB/s。GB10 の 273 GB/s に対して 68.4% で、
カーネルの効率ではなく読む量が効いている。

このうち lm_head は `tie_word_embeddings` を外して別テンソルにすれば、
単独で NVFP4 にできる。結果は同じ γ8 で 100.5 → **109.7 tok/s**（+9.2%）、
投機なし同士で 28.8 → **35.8 tok/s**（+24.3%）。

効かなかった打ち手も記録しておく。

- MoE カーネルを MARLIN に強制しても、非投機の単発は変わらなかった
- NVFP4 チェックポイントに `--quantization fp8` を重ねても、起動ログは
  `quantization=modelopt_fp4` のままで指定は無視される
- vision tower（0.59 GB）は text-only の decode では読まれない。
  `--language-model-only` を付けても速度は変わらなかった

## 次の課題

### 全層 4bit ＋ 日本語校正（試した結果、採用せず）

本レシピの重みは attention と共有 MLP を BF16 のまま残している。そこも含めて
すべて NVFP4 にし、日本語の校正データ 365 件と誤差補償つきの量子化（GPTQ 系）で
作り直した版を実測した。

| 指標 | 本レシピ | 全層 4bit 版 |
|---|---:|---:|
| 単発 tok/s | 109.7 | **122.3（+11.5%）** |
| 並列 C=32 合計 tok/s | 1,080.8 | **586.5（約半分）** |
| JNLI のペア差 | −1.23 / −1.48 pt | −1.64 pt |
| 5 指標の点推定 | すべて −2pt 以内 | すべて −2pt 以内 |

**質は保てたが並列で大きく落ちたため採用しなかった。** 単発では速く、日本語の
生成も破綻しない。

並列時だけ遅くなる理由は分かっていない。単発は帯域律速で、並列時は計算律速へ
移る可能性があるが、これは未検証の推測である。同じ測定方法で原因を説明して
低下を避けられれば、並列性能を保ったまま単発の +11.5% を得られる可能性がある。

### attention を FP8 にした版（質を最優先するなら）

全層 4bit で JNLI がやや落ちたので、attention（QKVO）だけ FP8 に戻し、
共有 MLP・routed expert・lm_head を NVFP4 にした版も作った。校正データは同じ
日本語 365 件。

| 指標 | 本レシピ | attention FP8 版 |
|---|---:|---:|
| 単発 tok/s | 109.7 | **117.5（+7.1%）** |
| 並列 C=32 合計 tok/s | 1,080.8 | 約半分 |
| JNLI のペア差（valid） | −1.23 / −1.48 pt | **−0.74 pt** |
| JNLI のペア差（独立セット） | −1.60 pt | **−1.20 pt** |
| 5 指標の点推定 | すべて −2pt 以内 | すべて −2pt 以内 |

**単発も質も本レシピを上回るが、並列が約半分なので採用していない。** バッチ用途では
合計スループットが効くため。単発しか使わず、質を最優先する場合はこちらの方が良い
可能性がある。重みは公開していない（作り方は本文の手順と同じで、attention だけ
FP8 target にする）。

attention を高い精度に戻すと質が戻り単発がわずかに落ちる、というトレードオフは
全層 4bit 版との比較にも現れている（JNLI −1.64 → −0.74 pt、単発 122.3 → 117.5）。

## ライセンス

Apache License 2.0 — `LICENSE` と `NOTICE` を参照。モデル本体の
利用には Gemma Terms of Use と禁止用途ポリシーが引き続き適用される。

## Tools used

Tools used — Claude Fable 5.1, GPT-6 (Astra), GLM-5.3-Flash. All code in this repository was written for this project.

## 謝辞

Google(Gemma 4 本体と MTP ドラフト)、NVIDIA(NVFP4 チェックポイントと
DGX Spark)、vLLM(サービングエンジン)に感謝する。
