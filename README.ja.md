# Gemma 4 26B-A4B on DGX Spark, v2

単一の DGX Spark�(��で Gemma 4 26B-A4B の NVFP4 ビルドを提供するための設定です。モデルの重みは v1 から変更していません。v1 も引き続き利用できます。git tag `v1` と image `tenhkspark/gemma-4-v2:v2` で利用できます。

## v1 から v2 への変更

| tok/s（特記がない限り） | v1 | v2 |
|---|---|---|
| コンテキスト長 | 32,768 | 262,144 |
| 初回トークンまでの時間、32k プロンプト | 12.4 s | 6.1 s（prefill-first profile） |
| 初回トークンまでの時間、128k プロンプト | 未対応 | 56 s（prefill-first profile） |
| 初回トークンまでの時間、250k プロンプト | 未対応 | 182 s（prefill-first profile） |
| デコード、日本語チャット、同時実行数 1 / 32 での tok/s | 45.9 / 616 | 62.9 / 850 |
| デコード、コーディング | 67.8 / 644 | 69.5 / 725 |
| デコード、ツール呼び出し | 54.6 / 193 | 61.4 / 272 |
| デコード、長文書 | 29.1 / 152 | 38.0 / 448 |
| デコード、構造化抽出（`GEMMA4_MTP=8` オプション） | 109.7 / 1,080.8（公開値） | 約 100 / 934 |
| 品質（ベンチマークと実利用での確認） | ベースライン | 同等 |
| コールドスタート | 約 4 min | 約 4 min |

バランス型プロファイルでの 2k / 8k / 30k プロンプトの初回トークンまでの時間は v1 と同じです（0.32 s / 1.57 s / 12.3 s）。安定性については、ルーター経由で短いリクエスト 32 件と長いリクエスト 4 件（最大 131k）を同時に 15 minutes 実行し、エラー 0 件、タイムアウト 0 件でした。

## アップグレード時の変更点

- Image: [tenhkspark/gemma-4-v2:v2](https://hub.docker.com/r/tenhkspark/gemma-4-v2:v2)
- Env files: `gemma4-v2.env` と、いずれか 1 つのプロファイル `gemma4-v2-balanced.env`（デフォルト）または `gemma4-v2-prefill-first.env`（長いプロンプト向け）
- Serve script: `gemma4-v2-serve.sh`、chat template `chat_template.jinja`
- Router（任意、複数ノード向け）: `tools/router.py` と `tools/router-v2.tsv`

```bash
docker pull tenhkspark/gemma-4-v2:v2
cp gemma4-v2*.env gemma4-v2-serve.sh chat_template.jinja ~/gemma4-spark/
cd ~/gemma4-spark && ./gemma4-v2-serve.sh --env gemma4-v2-balanced.env up
./gemma4-v2-serve.sh smoke
```

サーバーはポート 8890（`/v1/chat/completions`）で待ち受けます。Router: `python3 tools/router.py --config tools/router-v2.tsv --listen 0.0.0.0:8899`。`router-v2.tsv` のバランス型の上限は 32,768 トークンです。

## Speculative decoding（MTP）の設定

`GEMMA4_MTP=2` がデフォルトで、計測したすべてのワークロードで最速、またはそれに近い速度でした（上の表を参照）。

`GEMMA4_MTP=8`（`gemma4-v2-balanced-mtp8.env`）は、構造化抽出、テンプレートへの入力、ログの要約に適しています。270 個の異なるプロンプトで、単一ストリームでは約 100 tok/s、同時実行数 32 では 934 tok/s でした。自由形式のチャットでは遅く（単一ストリームで 48.9 tok/s に対して 62.9 tok/s）、コーディングのプロンプトでは同時実行数 8 と 32 で 8-10% 低下します。ワークロードごとに選択してください。変更する環境変数は 1 つです。バランス型プロファイルでは `GEMMA4_PREFIX_CACHE=0` を維持してください。

## 簡潔な応答を得る

Gemma 4 はデフォルトで冗長です。リクエスト側の 4 つの設定で応答を短くできます。

- 短いシステムプロンプト。例: `Answer in 3 sentences, no preamble.`
- `chat_template_kwargs: {"enable_thinking": false}` を指定すると、そのリクエストでは思考をオフにできます。
- `max_tokens` で応答の長さに上限を設定できます。
- 思考を有効にする場合は、`reasoning_effort` と `thinking_token_budget` で思考を短くできます。

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

このリクエストでのテスト回答は 47 tokens でした。

## 品質

`GEMMA4_MTP=2` のバランス型では、v1 との実利用比較はサンプルモードで合格し、125 問のベンチマークも同等でした（82.4% に対して 82.4%）。greedy-mode での比較 1 件は誤差の範囲内でしたが、より厳しい社内基準を下回りました。`GEMMA4_MTP=8` では、実利用の両モードが合格し、ベンチマークは 83.2% に対して 84.0% でした。

## 貢献者

tenhkspark。GLM-5.3 と GLM-5.3-Flash（Z.ai）が、分析・テストセット・ドキュメントの作成を支援しました。

ライセンス: `LICENSE` と `NOTICE` を参照してください。モデルの利用には Gemma Terms of Use が引き続き適用されます。
