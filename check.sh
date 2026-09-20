#!/bin/bash
# check.sh — prohibition scan over the whole gemma4-spark publish set.
#
#   bash check.sh        prints [PASS]/[FAIL] per check; exit 1 on any FAIL
#
# Scans every text file in this directory, this script included. Every
# pattern below is written with single-char [x] classes so that no banned
# literal ever appears inside this file itself — each regex still matches
# the real string (e.g. R[e]dHatAI matches the vendor handle).
#
# Checks:
#   1. retracted numbers/claims — the withdrawn figures and the withdrawn
#      quality verdict
#   2. third-party names — only Google, NVIDIA and vLLM may be named;
#      other org names, person names, blog names and judge-API names are
#      banned. The verbatim "Tools used" credit line is the single
#      exemption.
#   3. wording that points at other people's records/mistakes
#   4. the verbatim Tools-used line must be present at least once
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# The required credit line as a regex (bracketed so this file carries no
# banned literal). A line matching it is exempt from check 2.
TOOLS_RE='Tools used — C[l]aude F[a]ble 5\.1, G[P]T-6 \([A]stra\), G[L]M-5\.3-F[l]ash\. All code in this repository was written for this project\.'

# Check 1: withdrawn numbers and the withdrawn verdict.
RETRACTED_RE='674[.]6|477[.]9|110[.]4|[nN] ?= ?500|非[劣]性[^。]{0,8}[合]格|[合]格[^。]{0,8}非[劣]性|非[劣]性[^。]{0,8}(確[認]できた|確[認]した|確[認]できました|確[認]しました|成立)|非[劣]性[^。]{0,8}通[过]|非[劣]性[^。]{0,8}[^未不]确[认]|non-?inferiority (is |was |has |has been )?(passed|confirmed|established|achieved|met)'

# Check 2: banned names — other orgs, persons, blogs, judge APIs.
NAMES_RE='[A]nthropic|[O]penAI|\bM[e]ta\b|M[i]stral|M[i]crosoft|\bA[p]ple\b|\bA[m]azon\b|\bA[M]D\b|\bI[n]tel\b|Z\.?[aA][iI]\b|z[a]i-org|[mM]oonshot|D[e]epSeek|R[e]dHatAI|R[e]d Hat|b[g]-digitalservices|G[l]adiator|sbul[l]-dell|cklau[s]|majenti[k]|dhruvil23[7]|F[i]rworks|F[i]reworks|inclusio[n]AI|X[i]aomi|M[i]n[i]Max|A[l]ibaba|\bQ[w]en\b|\bC[l]ine\b|C[o]gnition|\bD[e]vin\b|[A]stra|C[l]aude|F[a]ble|G[P]T-[0-9]|\b[kK]imi\b|G[L]M-5|Q[i]ita|Z[e]nn|はて[な]|haten[a]|note[.]com|medium[.]com|substac[k]'

# Check 3: phrasing that points at other people's records.
OTHERS_RE='他[者]|よくある[誤間違ミ]|ありがちな[誤間違ミ]|他の[記]録|他の[報]告|先[行]の?記録|c[o]mmon [m]istake|others? (report|claim|believe|have found)'

NFAIL=0
verdict() { # <name> <PASS|FAIL> <detail>
  printf '[%-4s] %s — %s\n' "$2" "$1" "$3"
  [ "$2" = FAIL ] && NFAIL=$((NFAIL + 1))
  return 0
}

# all text files in the directory (binary files skipped)
FILES=()
while IFS= read -r f; do FILES+=("$f"); done < <(
  find "$HERE" -type f -not -path '*/.git/*' | sort | while IFS= read -r f; do
    grep -Iq . "$f" 2> /dev/null && printf '%s\n' "$f"
  done)
[ "${#FILES[@]}" -gt 0 ] || { echo "no text files under $HERE" >&2; exit 1; }
echo "check.sh: scanning ${#FILES[@]} text files under $HERE"

# ---- 1. retracted numbers/claims -------------------------------------------
hits="$(grep -HnE "$RETRACTED_RE" "${FILES[@]}" 2>/dev/null || true)"
if [ -z "$hits" ]; then
  verdict "retracted numbers" PASS "none present"
else
  printf '%s\n' "$hits" | sed "s|$HERE/||" | sed 's/^/    /'
  verdict "retracted numbers" FAIL "$(printf '%s\n' "$hits" | grep -c .) hit(s)"
fi

# ---- 2. banned names (Tools-used line exempt) -------------------------------
# filter: drop a hit only when its line carries the credit line AND no
# banned name survives once the verbatim credit text is removed
hits="$(grep -HnE "$NAMES_RE" "${FILES[@]}" 2>/dev/null \
  | python3 -c '
import re, sys
tools = re.compile(sys.argv[1]); names = re.compile(sys.argv[2])
for line in sys.stdin:
    m = re.match(r"(.*):([0-9]+):", line.rstrip("\n"))
    if not m:
        sys.stdout.write(line); continue
    f, n = m.group(1), int(m.group(2))
    try:
        src = open(f, encoding="utf-8").read().splitlines()[n - 1]
    except (OSError, IndexError):
        sys.stdout.write(line); continue
    if tools.search(src) and not names.search(tools.sub("", src)):
        continue
    sys.stdout.write(line)
' "$TOOLS_RE" "$NAMES_RE")"
if [ -z "$hits" ]; then
  verdict "banned names" PASS "only Google / NVIDIA / vLLM named"
else
  printf '%s\n' "$hits" | sed "s|$HERE/||" | sed 's/^/    /'
  verdict "banned names" FAIL "$(printf '%s\n' "$hits" | grep -c .) hit(s)"
fi

# file/dir names are part of the surface too
name_hits="$(printf '%s\n' "${FILES[@]}" | sed "s|$HERE/||" \
  | grep -E "$NAMES_RE" || true)"
if [ -z "$name_hits" ]; then
  verdict "banned names in filenames" PASS "none"
else
  printf '%s\n' "$name_hits" | sed 's/^/    name: /'
  verdict "banned names in filenames" FAIL "$(printf '%s\n' "$name_hits" | grep -c .) name(s)"
fi

# ---- 3. other-people wording ------------------------------------------------
hits="$(grep -HnE "$OTHERS_RE" "${FILES[@]}" 2>/dev/null || true)"
if [ -z "$hits" ]; then
  verdict "other-people wording" PASS "none present"
else
  printf '%s\n' "$hits" | sed "s|$HERE/||" | sed 's/^/    /'
  verdict "other-people wording" FAIL "$(printf '%s\n' "$hits" | grep -c .) hit(s)"
fi

# ---- 4. required credit line ------------------------------------------------
if grep -lE "$TOOLS_RE" "${FILES[@]}" 2>/dev/null | grep -q .; then
  verdict "Tools-used line" PASS "verbatim line present"
else
  verdict "Tools-used line" FAIL "required verbatim line missing"
fi

echo
[ "$NFAIL" = 0 ] && { echo "check.sh: PASS"; exit 0; }
echo "check.sh: $NFAIL FAIL item(s)"; exit 1
