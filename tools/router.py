#!/usr/bin/env python3
"""router.py -- small chat-completions router for gemma4 nodes.

Reads a TSV of backends and forwards /v1/chat/completions to the first healthy row that fits the request:

    profile<TAB>url<TAB>max_prompt_tokens<TAB>thinking

- profile: label, logged per request
- url: backend base, e.g. http://HOST:8890
- max_prompt_tokens: the request is eligible when the estimated prompt + requested output <= this
- thinking: on | off | any -- which requests the row accepts

Routes by estimated prompt size; falls back to a healthy backend.
GET /healthz reports backend health. The TSV is reloaded when its modification time changes.
Usage: python3 router.py --config router-v2.tsv --listen 0.0.0.0:8899
"""
import argparse
import http.client
import json
import math
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CHARS_PER_TOKEN = 3.5
CONNECT_TIMEOUT = float(os.environ.get("ROUTER_CONNECT_TIMEOUT", "5"))
TIMEOUT = float(os.environ.get("ROUTER_TIMEOUT", "900"))
HEALTH_PATH = os.environ.get("ROUTER_HEALTH_PATH", "/v1/models")
HEALTH_INTERVAL = float(os.environ.get("ROUTER_HEALTH_INTERVAL", "10"))
HEALTH_TIMEOUT = 5.0


def log(msg):
    sys.stderr.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} router {msg}\n")
    sys.stderr.flush()


class Config:
    """TSV rows, reloaded when the file's modification time changes."""

    def __init__(self, path):
        self.path = path
        self._cfg = (None, [])
        self.reload()

    def reload(self):
        rows = []
        with open(self.path) as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                cols = line.split("\t")
                if cols[0] == "profile":
                    continue
                if len(cols) != 4:
                    raise ValueError(f"{self.path}:{ln}: want 4 tab-separated columns, got {len(cols)}")
                profile, url, max_tok, think = cols
                url = url.rstrip("/")
                p = urllib.parse.urlsplit(url)
                if p.scheme not in ("http", "https") or not p.hostname:
                    raise ValueError(f"{self.path}:{ln}: bad url {url!r}")
                if think not in ("on", "off", "any"):
                    raise ValueError(f"{self.path}:{ln}: thinking must be on|off|any")
                rows.append({"profile": profile, "url": url, "https": p.scheme == "https",
                             "host": p.hostname, "port": p.port or (443 if p.scheme == "https" else 80),
                             "base": p.path.rstrip("/"), "max": int(max_tok), "thinking": think})
        if not rows:
            raise ValueError(f"{self.path}: no backend rows")
        self._cfg = (os.path.getmtime(self.path), rows)

    def get(self):
        mtime, rows = self._cfg
        try:
            if os.path.getmtime(self.path) != mtime:
                self.reload()
                log(f"config reloaded: {len(self._cfg[1])} rows")
        except Exception as e:
            log(f"config reload failed, keeping the old table: {e}")
        return self._cfg[1]


class Health:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = {}

    def mark(self, url, ok):
        with self._lock:
            prev = self._state.get(url)
            self._state[url] = ok
            return prev

    def ok(self, url):
        with self._lock:
            return self._state.get(url, True)

    def snapshot(self):
        with self._lock:
            return dict(self._state)


def _chars(o):
    if isinstance(o, str):
        return len(o)
    if isinstance(o, dict):
        return sum(_chars(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return sum(_chars(v) for v in o)
    return 0


def est_tokens(body):
    """Rough prompt size: characters over all message content / 3.5."""
    total = 0
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            total += _chars(m)
            continue
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            for p in c:
                total += len(p["text"]) if isinstance(p, dict) and isinstance(p.get("text"), str) else _chars(p)
        total += len(m.get("role") or "")
        total += _chars(m.get("tool_calls")) + _chars(m.get("function_call"))
    total += _chars(body.get("prompt"))
    return math.ceil(total / CHARS_PER_TOKEN)


def want_thinking(body):
    budget = body.get("thinking_token_budget")
    if budget is not None:
        try:
            if float(budget) > 0:
                return "on"
        except (TypeError, ValueError):
            if budget:
                return "on"
    eff = body.get("reasoning_effort")
    if eff is not None and str(eff).lower() != "none":
        return "on"
    kw = body.get("chat_template_kwargs")
    if isinstance(kw, dict):
        if "enable_thinking" in kw:
            return "on" if kw["enable_thinking"] else "off"
        if kw.get("thinking"):
            return "on"
    return "off"


def candidates(rows, health, est, think, reserve=0):
    """Backend order for one request."""
    match = [r for r in rows if r["thinking"] in ("any", think)]
    fits = [r for r in match if est + reserve <= r["max"]]
    order = fits or sorted(match, key=lambda r: r["max"], reverse=True)
    return [r for r in order if health.ok(r["url"])] + [r for r in order if not health.ok(r["url"])]


def _conn(row, timeout):
    cls = http.client.HTTPSConnection if row["https"] else http.client.HTTPConnection
    return cls(row["host"], row["port"], timeout=timeout)


def open_chat(row, body_bytes, headers):
    conn = _conn(row, CONNECT_TIMEOUT)
    try:
        conn.connect()
        conn.sock.settimeout(TIMEOUT)
        conn.request("POST", (row["base"] or "") + "/v1/chat/completions", body=body_bytes, headers=headers)
        return conn, conn.getresponse()
    except Exception:
        conn.close()
        raise


def _read1(resp, n=65536):
    return resp.read1(n) if hasattr(resp, "read1") else resp.read(n)


def _probe(row):
    conn = _conn(row, HEALTH_TIMEOUT)
    try:
        conn.request("GET", (row["base"] or "") + HEALTH_PATH)
        resp = conn.getresponse()
        resp.read()
        return resp.status < 400
    except Exception:
        return False
    finally:
        conn.close()


def _prober(cfg, health, interval, stop):
    while not stop.wait(interval):
        try:
            seen = set()
            for r in cfg.get():
                if r["url"] in seen:
                    continue
                seen.add(r["url"])
                ok = _probe(r)
                prev = health.mark(r["url"], ok)
                if prev is None or prev != ok:
                    log(f"backend {r['url']} {'up' if ok else 'down'}")
        except Exception as e:
            log(f"prober error: {e}")


def send_json(handler, status, obj):
    data = json.dumps(obj).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def stream_out(handler, resp, first, profile):
    handler.send_response(resp.status)
    handler.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
    handler.send_header("X-Router-Backend", profile)
    handler.send_header("Transfer-Encoding", "chunked")
    handler.end_headers()
    chunk = first
    while chunk:
        handler.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
        handler.wfile.flush()
        chunk = _read1(resp)
    handler.wfile.write(b"0\r\n\r\n")
    handler.wfile.flush()


def forward_chat(handler, cfg, health, body_bytes):
    try:
        body = json.loads(body_bytes)
        if not isinstance(body, dict):
            raise ValueError("object expected")
    except ValueError as e:
        return send_json(handler, 400, {"error": f"bad json: {e}"})
    est = est_tokens(body)
    reserve = body.get("max_completion_tokens") or body.get("max_tokens") or 0
    try:
        reserve = int(reserve)
    except (TypeError, ValueError):
        reserve = 0
    order = candidates(cfg.get(), health, est, want_thinking(body), reserve)
    if not order:
        return send_json(handler, 503, {"error": "no backend accepts this request"})
    headers = {"Content-Type": "application/json"}
    if handler.headers.get("Authorization"):
        headers["Authorization"] = handler.headers["Authorization"]
    last_err = "no backend tried"
    for row in order:
        try:
            conn, resp = open_chat(row, body_bytes, headers)
        except Exception as e:
            health.mark(row["url"], False)
            last_err = f"{row['url']}: {e}"
            continue
        try:
            first = _read1(resp)
            if resp.status >= 500:
                last_err = f"{row['url']}: HTTP {resp.status}"
                continue
            log(f"profile={row['profile']} est={est} status={resp.status}")
            return stream_out(handler, resp, first, row["profile"])
        except Exception as e:
            last_err = f"{row['url']}: {e}"
        finally:
            conn.close()
    return send_json(handler, 502, {"error": f"all backends failed: {last_err}"})


def make_handler(cfg, health):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            if self.path.split("?")[0] == "/v1/chat/completions":
                n = int(self.headers.get("Content-Length") or 0)
                return forward_chat(self, cfg, health, self.rfile.read(n))
            return send_json(self, 404, {"error": "not found"})

        def do_GET(self):
            path = self.path.split("?")[0]
            rows = cfg.get()
            if path == "/healthz":
                snap = health.snapshot()
                backends = [{"profile": r["profile"], "url": r["url"], "healthy": snap.get(r["url"], True)} for r in rows]
                return send_json(self, 200 if all(b["healthy"] for b in backends) else 503,
                                 {"ok": all(b["healthy"] for b in backends), "backends": backends})
            for row in [r for r in rows if health.ok(r["url"])] or rows:
                conn = _conn(row, HEALTH_TIMEOUT)
                try:
                    conn.request("GET", (row["base"] or "") + self.path)
                    resp = conn.getresponse()
                    data = resp.read()
                    self.send_response(resp.status)
                    self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                except Exception:
                    continue
                finally:
                    conn.close()
            return send_json(self, 502, {"error": "no backend reachable"})

    return Handler


def main():
    ap = argparse.ArgumentParser(description="length-based router for gemma4 nodes")
    ap.add_argument("--config", required=True, help="TSV: profile, url, max_prompt_tokens, thinking")
    ap.add_argument("--listen", default="0.0.0.0:8899", help="host:port")
    args = ap.parse_args()
    host, _, port = args.listen.rpartition(":")
    cfg, health, stop = Config(args.config), Health(), threading.Event()
    threading.Thread(target=_prober, args=(cfg, health, HEALTH_INTERVAL, stop), daemon=True).start()
    srv = ThreadingHTTPServer((host or "0.0.0.0", int(port)), make_handler(cfg, health))
    srv.daemon_threads = True
    log(f"listening on {args.listen}, config {args.config} ({len(cfg.get())} backends)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()


if __name__ == "__main__":
    main()
