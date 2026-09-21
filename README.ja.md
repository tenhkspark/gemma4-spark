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
- コンテナイメージ — vLLM 上流の `docker/Dockerfile` をビルド
  (コマンドと引数は `BRING-UP.md` §2)

## クイックスタート

1. **重みを取得**（`huggingface-cli download tenhkspark/gemma-4-26B-A4B-NVFP4-lmhead --local-dir ./gemma4-lmhead`）。検証用 manifest は
   13 ファイル / 19,240,726,248 B / md5 `571932348835310ce77799f70a4e9814`。
2. **コンテナを用意**(`BRING-UP.md` §2)。
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
生成も破綻しない。並列だけが遅い理由は分かっていない（単発は帯域律速、並列は
計算律速に入るため、という見方はあるが未検証）。

同じ測り方で並列の落ち込みを説明し、避けられるなら、単発の +11.5% を保ったまま
並列も取れる可能性がある。

## ライセンス

Apache License 2.0 — `LICENSE` と `NOTICE` を参照。モデル本体の
利用には Gemma Terms of Use と禁止用途ポリシーが引き続き適用される。

## Tools used

Tools used — Claude Fable 5.1, GPT-6 (Astra), GLM-5.3-Flash. All code in this repository was written for this project.

## 謝辞

Google(Gemma 4 本体と MTP ドラフト)、NVIDIA(NVFP4 チェックポイントと
DGX Spark)、vLLM(サービングエンジン)に感謝する。
