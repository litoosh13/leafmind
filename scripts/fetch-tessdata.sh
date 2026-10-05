#!/bin/sh
# Downloads the Tesseract language files leafmind-ocr uses (about 50 MB) into <dir>: English, German, Persian
# and Arabic from tessdata_best, and osd (script detection) from tessdata, at fixed commits, and checks every
# file's SHA-256. Files already present with the right checksum are not downloaded again.
# Sources and licences: THIRD_PARTY.md. Usage: scripts/fetch-tessdata.sh <dir>
set -eu
dir=${1:?usage: $0 <dir>}
mkdir -p "$dir"

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

# repository  commit  file  sha256
while read -r repo commit file sum; do
    out="$dir/$file"
    if [ -f "$out" ] && [ "$(sha256 "$out")" = "$sum" ]; then
        echo "ok (already there): $out"
        continue
    fi
    echo "downloading $repo/$file"
    curl -fL --retry 3 -o "$out.part" "https://github.com/tesseract-ocr/$repo/raw/$commit/$file"
    if [ "$(sha256 "$out.part")" != "$sum" ]; then
        echo "checksum mismatch for $out" >&2
        rm -f "$out.part"
        exit 1
    fi
    mv "$out.part" "$out"
    echo "ok: $out"
done <<'LIST'
tessdata_best e12c65a915945e4c28e237a9b52bc4a8f39a0cec eng.traineddata 8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba
tessdata_best e12c65a915945e4c28e237a9b52bc4a8f39a0cec deu.traineddata 8407331d6aa0229dc927685c01a7938fc5a641d1a9524f74838cdac599f0d06e
tessdata_best e12c65a915945e4c28e237a9b52bc4a8f39a0cec fas.traineddata 99e420969b5ddd2cb135b416316a7ed417c59c4faf9e0d28941348f6448114df
tessdata_best e12c65a915945e4c28e237a9b52bc4a8f39a0cec ara.traineddata ab9d157d8e38ca00e7e39c7d5363a5239e053f5b0dbdb3167dde9d8124335896
tessdata ced78752cc61322fb554c280d13360b35b8684e4 osd.traineddata e19f2ae860792fdf372cf48d8ce70ae5da3c4052962fe22e9de1f680c374bb0e
LIST
echo "done: $dir"
