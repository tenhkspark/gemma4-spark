#!/usr/bin/env python3
"""bench-cell.py -- one C=1 speed cell for the Gemma 4 bring-up check.

One warmup request, then one document of each of the 3 workload types
(json / fin / log), temperature 0, max_tokens 512, sequential
(concurrency 1). Prints per-request headline tok/s (= completion_tokens
/ wall, TTFT included) and the mean of the 3.

Usage: python3 bench-cell.py [base-url]   (default http://127.0.0.1:8890)
"""
import json
import sys
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8890").rstrip("/")

# Fictional JA business docs (doc index 0 of each type).
DOCS = [
    ("json",
     "次のチケットから JSON を抽出してください。キーは customer, product, issue, priority, due です。"
     "値は本文の語句をそのまま使ってください。\n"
     "チケット: 顧客「株式会社ミナト運輸（架空）」より、製品「配車管理クラウド v3.2」で"
     "「配車表の PDF 出力が 2026-09-18 以降空白になる」という報告。"
     "優先度は高、対応期限は 2026-09-25。担当は東部サポート。"
     "備考: ブラウザは Chrome 130、再現率 100%。\n"
     "出力は JSON のみ。"),
    ("fin",
     "次の財務諸表テンプレートの空欄を、与えられた数値で埋めて全文を出力してください。\n"
     "テンプレート:\n"
     "【損益計算書（株式会社キタガワ建材・架空・第41期・単位: 千円）】\n"
     "売上高: ____\n売上原価: ____\n売上総利益: ____\n"
     "販売費及び一般管理費: ____\n営業利益: ____\n"
     "数値: 売上高 482,300 / 売上原価 301,150 / 販売費及び一般管理費 122,800。"
     "売上総利益と営業利益は計算してください。"),
    ("log",
     "次の監視ログを要約してください。要約には発生時刻、ホスト名、事象、対応を本文の語句のまま含めてください。\n"
     "ログ:\n"
     "2026-09-20 03:12:44 host=db-primary-02 event=replication_lag lag=48s threshold=30s action=alert_sent\n"
     "2026-09-20 03:13:10 host=db-primary-02 event=replication_lag lag=71s threshold=30s action=failover_started\n"
     "2026-09-20 03:14:02 host=db-replica-01 event=promoted role=primary action=dns_switch\n"
     "2026-09-20 03:14:30 host=app-web-07 event=reconnect target=db-replica-01 result=ok\n"
     "要約は 3 行。"),
]


def chat(model, prompt, max_tokens):
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read().decode())
    wall = time.monotonic() - t0
    ct = (d.get("usage") or {}).get("completion_tokens") or 0
    fin = (d.get("choices") or [{}])[0].get("finish_reason")
    return ct, wall, fin


def main():
    model = json.loads(urllib.request.urlopen(BASE + "/v1/models",
                                              timeout=10).read())["data"][0]["id"]
    print(f"bench-cell: base={BASE} model={model}")
    ct, wall, _ = chat(model, "ウォームアップ。OK とだけ返答して。", 32)
    print(f"warmup: ct={ct} wall={wall:.1f}s (excluded)")
    hls = []
    for tag, prompt in DOCS:
        ct, wall, fin = chat(model, prompt, 512)
        hl = ct / wall
        hls.append(hl)
        print(f"{tag}: ct={ct} wall={wall:.2f}s headline={hl:.1f} tok/s fin={fin}")
    print(f"mean of 3 = {sum(hls) / len(hls):.1f} tok/s")


if __name__ == "__main__":
    main()
