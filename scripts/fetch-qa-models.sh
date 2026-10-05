#!/bin/sh
# Downloads the two models leafmind-qa needs (about 680 MB) into <dir>/gte-embed and <dir>/gte-reranker,
# and with "accurate" also accurate mode's reranker (about 2.4 GB) into <dir>/qwen3-reranker, from fixed
# Hugging Face revisions, and checks every file's SHA-256. Files already present with the right checksum are
# not downloaded again. Sources and licences: THIRD_PARTY.md.
# Usage: scripts/fetch-qa-models.sh <dir> [accurate]
set -eu
dir=${1:?usage: $0 <dir> [accurate]}
accurate=${2:-}

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

# folder  repository  revision  file-in-repo  sha256  [saved name]
while read -r folder repo rev file sum name; do
    [ "$folder" = "qwen3-reranker" ] && [ "$accurate" != "accurate" ] && continue
    out="$dir/$folder/${name:-$(basename "$file")}"
    mkdir -p "$dir/$folder"
    if [ -f "$out" ] && [ "$(sha256 "$out")" = "$sum" ]; then
        echo "ok (already there): $out"
        continue
    fi
    echo "downloading $repo/$file"
    curl -fL --retry 3 -o "$out.part" "https://huggingface.co/$repo/resolve/$rev/$file"
    if [ "$(sha256 "$out.part")" != "$sum" ]; then
        echo "checksum mismatch for $out" >&2
        rm -f "$out.part"
        exit 1
    fi
    mv "$out.part" "$out"
    echo "ok: $out"
done <<'LIST'
gte-embed onnx-community/gte-multilingual-base 2edbf5e672aab465f9ed4c154a8b61791c082c69 onnx/model_int8.onnx ab2bd164ebd8ca9003dc49a981b611e849b5d326f504c8873ba76e07fa6c0082
gte-embed onnx-community/gte-multilingual-base 2edbf5e672aab465f9ed4c154a8b61791c082c69 tokenizer.json 3a56def25aa40facc030ea8b0b87f3688e4b3c39eb8b45d5702b3a1300fe2a20
gte-reranker onnx-community/gte-multilingual-reranker-base ee64367e35a2db0da46bb6497e13a18f8bd585cb onnx/model_int8.onnx ccf51dba7f8aa9205753761cfaa68c55f741792501463a3bf25d7e5bcdac7c35
gte-reranker onnx-community/gte-multilingual-reranker-base ee64367e35a2db0da46bb6497e13a18f8bd585cb tokenizer.json 3ffb37461c391f096759f4a9bbbc329da0f36952f88bab061fcf84940c022e98
qwen3-reranker litoo13/leafmind-qwen3-reranker-0.6b 18f5fcc209837acaf0f702b03006220af73cb623 model.onnx ec9d799e3a241bf06a5ceb0a1efb7d4a1645c19bd57ceed81ce89af4de2dde3b
qwen3-reranker litoo13/leafmind-qwen3-reranker-0.6b 18f5fcc209837acaf0f702b03006220af73cb623 model.onnx.data 22ab01e5d02189a8a4eac7f9da1ac027ad8f064635390da8797269f805399dda
qwen3-reranker litoo13/leafmind-qwen3-reranker-0.6b 18f5fcc209837acaf0f702b03006220af73cb623 tokenizer.json aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
LIST
echo "done: $dir"
