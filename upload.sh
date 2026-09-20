#!/bin/bash
# upload.sh — stage the gemma4-spark checkpoint + card and, only with an
# explicit flag, upload them to HF.
#
#   upload.sh --ckpt /path/to/untied-ckpt --repo <owner/name>   # dry-run
#   upload.sh --meta-only --repo <owner/name>                  # card/meta only
#   upload.sh ... --publish                                  # the real thing
#
# Dry-run is the default: it runs every local integrity check, rebuilds the
# manifests, and prints the upload calls a real run would make. A real upload
# happens ONLY when both hold:
#   --publish   on the command line (the explicit go flag)
#   a token that huggingface_hub resolves (HF_TOKEN env or `hf auth login`)
# No token value is ever printed, written, or stored by this script.
#
# Integrity gate — all of it must pass before anything is uploaded:
#   1. required checkpoint files present, >= 1 *.safetensors shard
#   2. config.json carries "tie_word_embeddings": false — the property that
#      defines this derivative; uploading a still-tied ckpt is a fail
#   3. payload file count and total bytes (weights ~19.2 GB; hard band
#      17-22 GB) match the expected profile
#   4. MD5SUMS manifest built over the payload, then verified by re-hashing
#      every file against it
#
# Outputs written next to this script (and uploaded with the repo):
#   files.tsv   arcname<TAB>bytes for every payload file
#   MD5SUMS     md5  arcname over the same payload
#
# --meta-only re-hashes and uploads just the card/metadata files (no --ckpt
# needed); checkpoint rows in the manifests are carried through untouched.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

CKPT=""; REPO=""; META=0; PUBLISH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --ckpt)      CKPT=$2; shift 2 ;;
    --repo)      REPO=$2; shift 2 ;;
    --meta-only) META=1; shift ;;
    --publish)   PUBLISH=1; shift ;;
    --dry-run)   shift ;;
    -h|--help)   sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ -n "$REPO" ] || { sed -n '2,30p' "$0" >&2; exit 2; }
if [ "$META" = 0 ]; then
  [ -n "$CKPT" ] || { sed -n '2,30p' "$0" >&2; exit 2; }
  [ -d "$CKPT" ] || { echo "FAIL: no checkpoint dir: $CKPT" >&2; exit 1; }
fi
DRY=1; [ "$PUBLISH" = 1 ] && DRY=0

echo "== upload.sh: ckpt=${CKPT:-(none)} repo=$REPO \
$([ "$META" = 1 ] && echo META-ONLY || echo FULL) \
$([ "$DRY" = 1 ] && echo DRY-RUN || echo PUBLISH)"

# ---- 1. checkpoint contents ----------------------------------------------
if [ "$META" = 0 ]; then
  echo "-- 1. checkpoint contents"
  missing=0
  for f in config.json tokenizer.json tokenizer_config.json; do
    [ -f "$CKPT/$f" ] || { echo "   MISSING $f"; missing=1; }
  done
  ls "$CKPT"/*.safetensors > /dev/null 2>&1 \
    || { echo "   MISSING *.safetensors"; missing=1; }
  for f in hf_quant_config.json model.safetensors.index.json \
           generation_config.json chat_template.jinja processor_config.json; do
    [ -f "$CKPT/$f" ] || echo "   warn: no $f"
  done
  if [ -f "$CKPT/config.json" ]; then
    grep -q '"tie_word_embeddings"[[:space:]]*:[[:space:]]*false' \
      "$CKPT/config.json" \
      || { echo "   MISSING tie_word_embeddings=false in config.json"; missing=1; }
  fi
  [ "$missing" = 0 ] || { echo "FAIL: checkpoint incomplete" >&2; exit 1; }
fi

# ---- 2. payload list (kind<TAB>arcname<TAB>bytes<TAB>src) ------------------
PAYLOAD="$(mktemp "${TMPDIR:-/tmp}/gemma4-payload.XXXXXX")"
trap 'rm -f "$PAYLOAD"' EXIT
python3 - "$HERE" "$CKPT" "$META" > "$PAYLOAD" <<'PY'
import os, sys
here, ckpt, meta_only = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
# README.md on HF is the model card (HF convention), so MODEL-CARD.md
# maps onto it. The publish-dir README.md is the same content in repo
# layout and is not uploaded under its own name.
META = [("MODEL-CARD.md", "README.md"), ("README.ja.md", "README.ja.md"),
        ("README.ko.md", "README.ko.md"), ("README.zh.md", "README.zh.md"),
        ("BRING-UP.md", "BRING-UP.md"),
        ("SERVING-NOTES-2026-09-20.ja.md", "SERVING-NOTES-2026-09-20.ja.md"),
        ("SERVING-NOTES-2026-09-20.ko.md", "SERVING-NOTES-2026-09-20.ko.md"),
        ("SERVING-NOTES-2026-09-20.zh.md", "SERVING-NOTES-2026-09-20.zh.md"),
        ("serve.sh", "serve.sh"), ("gemma4.env", "gemma4.env"),
        ("gemma4.small.env", "gemma4.small.env"),
        ("untie-lmhead-fp8.py", "untie-lmhead-fp8.py"),
        ("bench-cell.py", "bench-cell.py"), ("check.sh", "check.sh"),
        ("LICENSE", "LICENSE"), ("NOTICE", "NOTICE")]
rows = []
if not meta_only:
    for f in sorted(os.listdir(ckpt)):
        p = os.path.join(ckpt, f)
        if os.path.isfile(p):
            rows.append(("ckpt", f, os.path.getsize(p), p))
for loc, arc in META:
    p = os.path.join(here, loc)
    if os.path.isfile(p):
        rows.append(("meta", arc, os.path.getsize(p), p))
    else:
        print(f"warn: no {loc}", file=sys.stderr)
for k, a, b, s in rows:
    print(f"{k}\t{a}\t{b}\t{s}")
PY
NFILES=$(wc -l < "$PAYLOAD" | tr -d ' ')
[ "$NFILES" -gt 0 ] || { echo "FAIL: empty payload" >&2; exit 1; }

# ---- 3. file count / byte count ------------------------------------------
echo "-- 3. payload profile: $NFILES files"
if [ "$META" = 0 ]; then
  BYTES=$(awk -F'\t' '$1=="ckpt"{s+=$3} END{printf "%.0f", s}' "$PAYLOAD")
  awk -v t="$BYTES" 'BEGIN{printf "   ckpt bytes: %.1f GB\n", t/1e9}'
  awk -v t="$BYTES" 'BEGIN{exit !(t>=17e9 && t<=22e9)}' \
    || { echo "FAIL: ckpt bytes outside expected 17-22 GB band" >&2; exit 1; }
fi

# ---- 4. manifest build + verify -------------------------------------------
echo "-- 4. manifest (md5)"
python3 - "$HERE" "$PAYLOAD" "$META" <<'PY' || { echo "FAIL: manifest verification" >&2; exit 1; }
import hashlib, os, sys
here, payload_tsv, meta_only = sys.argv[1], sys.argv[2], sys.argv[3] == "1"

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

rows = []   # (arcname, bytes, md5)
for line in open(payload_tsv, encoding="utf-8"):
    kind, arc, size, src = line.rstrip("\n").split("\t")
    rows.append((arc, int(size), md5(src)))

files_tsv = os.path.join(here, "files.tsv")
md5sums   = os.path.join(here, "MD5SUMS")

if meta_only and os.path.isfile(files_tsv) and os.path.isfile(md5sums):
    # keep existing ckpt rows; replace only the rows this run produced
    arcs = {a for a, _, _ in rows}
    old_f = [l for l in open(files_tsv, encoding="utf-8")
             if l.split("\t")[0] not in arcs]
    old_m = [l for l in open(md5sums, encoding="utf-8")
             if len(l.split("  ", 1)) == 2
             and l.split("  ", 1)[1].strip() not in arcs]
    with open(files_tsv, "w", encoding="utf-8") as fh:
        fh.writelines(sorted(old_f + [f"{a}\t{b}\n" for a, b, _ in rows]))
    with open(md5sums, "w", encoding="utf-8") as fh:
        fh.writelines(sorted(old_m + [f"{m}  {a}\n" for a, _, m in rows]))
    print(f"   manifests: {len(rows)} meta row(s) updated, ckpt rows kept")
else:
    if meta_only:
        print("   warn: no prior manifest; writing meta rows only")
    with open(files_tsv, "w", encoding="utf-8") as fh:
        for a, b, _ in sorted(rows):
            fh.write(f"{a}\t{b}\n")
    with open(md5sums, "w", encoding="utf-8") as fh:
        for a, _, m in sorted(rows):
            fh.write(f"{m}  {a}\n")
    print(f"   manifests: {len(rows)} file(s) written")

# verify: re-hash every payload file against the manifest just written
man = {}
for l in open(md5sums, encoding="utf-8"):
    if "  " in l:
        m, a = l.rstrip("\n").split("  ", 1)
        man[a] = m
bad = 0
for line in open(payload_tsv, encoding="utf-8"):
    kind, arc, size, src = line.rstrip("\n").split("\t")
    if arc not in man:
        print(f"   MISSING manifest row: {arc}"); bad = 1; continue
    if md5(src) != man[arc]:
        print(f"   MD5 MISMATCH: {arc}"); bad = 1
if not meta_only:
    want = len(rows)
    have = sum(1 for line in open(files_tsv, encoding="utf-8"))
    if have != want:
        print(f"   files.tsv row count {have} != payload {want}"); bad = 1
sys.exit(1 if bad else 0)
PY
echo "   verify: all md5 match"

# ---- 5. upload plan --------------------------------------------------------
echo "-- 5. upload plan"
if [ "$DRY" = 1 ]; then
  [ "$META" = 0 ] \
    && echo "   DRY$ upload_folder $CKPT -> $REPO (repo_type=model)"
  awk -F'\t' -v repo="$REPO" \
    '$1=="meta"{print "   DRY$ upload_file " $4 " -> " repo ":" $2}' \
    "$PAYLOAD"
  echo "   DRY$ upload_file $HERE/files.tsv -> $REPO:files.tsv"
  echo "   DRY$ upload_file $HERE/MD5SUMS -> $REPO:MD5SUMS"
  echo "   (set --publish and provide a resolvable HF token to upload)"
  echo "upload.sh: dry-run done"
  exit 0
fi

# ---- real upload -----------------------------------------------------------
python3 - <<'PY' || { echo "FAIL: no HF token (HF_TOKEN or hf auth login)" >&2; exit 1; }
import sys
import huggingface_hub
sys.exit(0 if huggingface_hub.get_token() else 1)
PY
python3 - "$REPO" <<'PY' || { echo "FAIL: token rejected (whoami)" >&2; exit 1; }
import sys
from huggingface_hub import HfApi
HfApi().whoami()
PY

if [ "$META" = 0 ]; then
  echo "   upload $CKPT (weights)"
  python3 - "$CKPT" "$REPO" <<'PY' || { echo "FAIL: weights upload" >&2; exit 1; }
import sys
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path=sys.argv[1], repo_id=sys.argv[2],
                      repo_type="model",
                      ignore_patterns=["*.log", ".DS_Store", ".cache/**",
                                       "__pycache__/**"])
PY
fi
while IFS=$'\t' read -r kind arc _bytes src; do
  [ "$kind" = meta ] || continue
  echo "   upload $src -> $arc"
  python3 - "$src" "$arc" "$REPO" <<'PY' || { echo "FAIL: upload $arc" >&2; exit 1; }
import sys
from huggingface_hub import HfApi
src, arc, repo = sys.argv[1:4]
HfApi().upload_file(path_or_fileobj=src, path_in_repo=arc, repo_id=repo,
                    repo_type="model",
                    commit_message=f"gemma4-spark card: {arc}")
PY
done < "$PAYLOAD"
for f in files.tsv MD5SUMS; do
  echo "   upload $HERE/$f"
  python3 - "$HERE/$f" "$f" "$REPO" <<'PY' || { echo "FAIL: upload $f" >&2; exit 1; }
import sys
from huggingface_hub import HfApi
src, arc, repo = sys.argv[1:4]
HfApi().upload_file(path_or_fileobj=src, path_in_repo=arc, repo_id=repo,
                    repo_type="model",
                    commit_message=f"gemma4-spark manifest: {arc}")
PY
done
echo "upload.sh: upload done"
