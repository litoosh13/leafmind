#!/bin/sh
# Downloads the two models leafmind-qa needs (vocabulary pruned for English, German, Persian and Arabic) into
# <dir>/gte-embed and <dir>/gte-reranker, from fixed Hugging Face revisions, and checks every file's SHA-256: the
# int8 files for the native build (about 345 MB), or with "web" the q4f16 files for the browser build (about 355 MB).
# With "fill": leafmind-fill's fine-tuned model (<dir>/fill/fill.cact, 63 MB) and Needle's browser engine
# (<dir>/needle-web/needle.js and needle.wasm, 1 MB); leafmind-fill also needs gte-embed from the first or second form.
# With "needle-<platform>": Needle's native engine 3.2.0 for leafmind-fill, unpacked from Cactus's wheel into
# <dir>/needle/ (libneedle3.dylib, .so or .dll, about 1.5 MB). Platforms: macos-arm64, macos-x86_64, linux-x86_64,
# linux-arm64, linux-musl-x86_64, linux-musl-arm64, windows-x86_64, windows-arm64 (unpacking needs `unzip`).
# Files already present with the right checksum are not downloaded again. Sources and licences: THIRD_PARTY.md.
# Usage: scripts/fetch-qa-models.sh <dir> [web | fill | needle-<platform>]
set -eu
dir=${1:?usage: $0 <dir> [web | fill | needle-<platform>]}
want=${2:-native}

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

# build  folder  repository  revision  file-in-repo  sha256  [saved name]
found=0
while read -r build folder repo rev file sum name; do
    [ "$build" = "$want" ] || continue
    found=1
    out="$dir/$folder/${name:-$(basename "$file")}"
    mkdir -p "$dir/$folder"
    if [ -f "$out" ] && [ "$(sha256 "$out")" = "$sum" ]; then
        echo "ok (already there): $out"
        case "$out" in *.whl) unzip -o -j -q "$out" 'needle/libneedle3.*' -d "$dir/$folder" ;; esac
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
    case "$out" in *.whl) unzip -o -j -q "$out" 'needle/libneedle3.*' -d "$dir/$folder" && echo "unpacked into $dir/$folder" ;; esac
done <<'LIST'
native gte-embed litoo13/leafmind-gte-multilingual-4lang 466fd26d9e751de3a57adc3ae222115865ca8cc4 gte-embed/model_int8.onnx 5668afdae8b748de54ef85b7a13a711005ccf2a7a08806159f78b30c5f722cbf
native gte-embed litoo13/leafmind-gte-multilingual-4lang 466fd26d9e751de3a57adc3ae222115865ca8cc4 gte-embed/tokenizer.json c1bafc0e9c74e6cefcd032df267ce8603c7a288a771e517d9c8823e0a7de8a37
native gte-reranker litoo13/leafmind-gte-multilingual-4lang 466fd26d9e751de3a57adc3ae222115865ca8cc4 gte-reranker/model_int8.onnx 4e58a5fd59990fd78afc8a509f27d4f799d8b8723ceea00e0a07fd6f316fb542
native gte-reranker litoo13/leafmind-gte-multilingual-4lang 466fd26d9e751de3a57adc3ae222115865ca8cc4 gte-reranker/tokenizer.json e26ac9905f59291e2540981b82ca2e1022e9699b2456324f66ddba4804272847
web gte-embed litoo13/leafmind-gte-multilingual-4lang a789eef86239fee01ba034413e5af09291f28506 gte-embed/model_q4f16.onnx aa9ae652c7f6effbfc7aa57465f6072256926c944922dee3845cec456168864f
web gte-embed litoo13/leafmind-gte-multilingual-4lang a789eef86239fee01ba034413e5af09291f28506 gte-embed/tokenizer.json c1bafc0e9c74e6cefcd032df267ce8603c7a288a771e517d9c8823e0a7de8a37
web gte-reranker litoo13/leafmind-gte-multilingual-4lang a789eef86239fee01ba034413e5af09291f28506 gte-reranker/model_q4f16.onnx 8514babe9d028c6314a0ff8d6495bb778b55f54d7d4d3928589314a32c833188
web gte-reranker litoo13/leafmind-gte-multilingual-4lang a789eef86239fee01ba034413e5af09291f28506 gte-reranker/tokenizer.json e26ac9905f59291e2540981b82ca2e1022e9699b2456324f66ddba4804272847
fill fill litoo13/leafmind-fill-needle c52f3d0a11f38e6a01d094090501a35c90f2882b fill.cact 6198a4579107fc6b0546816a547f6c8594979ba2ebe33a4a4b2c8643536c35fd
fill needle-web Cactus-Compute/needle3 d6c7d6a2e9655505e0f8e23339356dd1705900eb wasm/needle.js a675c66f506a1b6f922d852c0cc0b82a843ce9d0fab6a80a408106f516df126f
fill needle-web Cactus-Compute/needle3 d6c7d6a2e9655505e0f8e23339356dd1705900eb wasm/needle.wasm 7b73d09c17923bf81d6bdb82e19b3c343b8994f83d512ddec52eec9f6b029802
needle-macos-arm64 needle Cactus-Compute/needle3 d6c7d6a2e9655505e0f8e23339356dd1705900eb python/cactus_needle-3.2.0-py3-none-macosx_11_0_arm64.whl 3b0887a43cd6e9a99009fabf35b231c11bb3a978ab8af94b8119b2eb76e19832
needle-macos-x86_64 needle Cactus-Compute/needle3 d6c7d6a2e9655505e0f8e23339356dd1705900eb python/cactus_needle-3.2.0-py3-none-macosx_11_0_x86_64.whl d2a60d538b138ecc5ac5a9b4d47ba8d078131d0af9e20e94cceb9329894d0f96
needle-linux-arm64 needle Cactus-Compute/needle3 d6c7d6a2e9655505e0f8e23339356dd1705900eb python/cactus_needle-3.2.0-py3-none-manylinux2014_aarch64.whl 80871585c84a277eff24b43a9366958996be38f6f920c9d1f8a0d2a07f3ae51f
needle-linux-x86_64 needle Cactus-Compute/needle3 d6c7d6a2e9655505e0f8e23339356dd1705900eb python/cactus_needle-3.2.0-py3-none-manylinux2014_x86_64.whl 0e8a3bce4e52968ee14e8e9d47098e65ed06dd5b7c18cef3a7fa1375756736f9
needle-linux-musl-arm64 needle Cactus-Compute/needle3 d6c7d6a2e9655505e0f8e23339356dd1705900eb python/cactus_needle-3.2.0-py3-none-musllinux_1_2_aarch64.whl 739dedd88473223720dd1fb8264cba6733d4bec39fec325bcfc0fb3c6c7df2bf
needle-linux-musl-x86_64 needle Cactus-Compute/needle3 d6c7d6a2e9655505e0f8e23339356dd1705900eb python/cactus_needle-3.2.0-py3-none-musllinux_1_2_x86_64.whl b91a479d0bb3a2d6a5b70e8b58db063ec5ae126c73a5e1a72dc9f9dd24e802fc
needle-windows-x86_64 needle Cactus-Compute/needle3 d6c7d6a2e9655505e0f8e23339356dd1705900eb python/cactus_needle-3.2.0-py3-none-win_amd64.whl 0bd41dc812a4b7e504f6ba294de65a363a51b031f154d24aac748b07249dc952
needle-windows-arm64 needle Cactus-Compute/needle3 d6c7d6a2e9655505e0f8e23339356dd1705900eb python/cactus_needle-3.2.0-py3-none-win_arm64.whl 15000affa669b7e84e6303617312e04650a05e9c71e856bab075a0ad36ebe8cf
LIST
[ "$found" = 1 ] || { echo "unknown choice: $want (see the comment at the top of $0)" >&2; exit 1; }
echo "done: $dir"
