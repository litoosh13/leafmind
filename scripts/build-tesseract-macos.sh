#!/bin/sh
# Builds Tesseract 5.5.3 with Leptonica 1.87.0 linked in, as one universal library (Apple silicon and Intel) that
# runs on macOS 11 and newer: <dir>/libtesseract.5.dylib, with the two licences (LICENSE-tesseract, Apache-2.0;
# LICENSE-leptonica, BSD-2-Clause). leafmind-ocr gives Tesseract raw pixels, so the build
# leaves out every image format, libarchive and libcurl; the library needs nothing but macOS itself. Its install
# name is @loader_path/libtesseract.5.dylib and it is signed ad hoc (an app signs it with its own identity).
# Downloads the two source releases (checked by SHA-256). Needs Xcode's command line tools, cmake and ninja
# (`brew install cmake ninja`, or `pip install cmake ninja`). Sources and licences: THIRD_PARTY.md.
# Usage: scripts/build-tesseract-macos.sh <dir> [lowest macOS version, default 11.0]
set -eu
dir=${1:?usage: $0 <dir> [lowest macOS version]}
min=${2:-11.0}
mkdir -p "$dir"
dir=$(cd "$dir" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

fetch() { # url sha256 folder
    curl -fsSL --retry 3 -o "$work/src.tgz" "$1"
    [ "$(shasum -a 256 "$work/src.tgz" | cut -d' ' -f1)" = "$2" ] || { echo "checksum mismatch: $1" >&2; exit 1; }
    tar xzf "$work/src.tgz" -C "$work"
}
fetch https://github.com/DanBloomberg/leptonica/releases/download/1.87.0/leptonica-1.87.0.tar.gz \
    c73363397f96eb1295602bf44d708a994ad42046c791bf03ea0505d829bdb6a7
fetch https://github.com/tesseract-ocr/tesseract/archive/refs/tags/5.5.3.tar.gz \
    9218e62793116d42a9f6d14cd9348518b27f382096eea3d0f2d1a24616bb5884

# Only macOS's own libraries: Homebrew's (or MacPorts') must not be found.
common="-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=$min
    -DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew;/usr/local;/opt/local -DCMAKE_FIND_FRAMEWORK=NEVER"

for arch in arm64 x86_64; do
    prefix="$work/$arch"
    # Tesseract picks its SIMD code (NEON or SSE/AVX) by the target processor, which CMake only takes from us
    # when cross-compiling; its one test program (Leptonica's TIFF support) is answered "no" (exit code 1).
    target="-DCMAKE_OSX_ARCHITECTURES=$arch -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_PROCESSOR=$arch"
    # shellcheck disable=SC2086
    cmake -S "$work/leptonica-1.87.0" -B "$work/build-lept-$arch" $common $target \
        -DCMAKE_INSTALL_PREFIX="$prefix" -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_PROG=OFF -DSW_BUILD=OFF -DENABLE_ZLIB=OFF -DENABLE_PNG=OFF -DENABLE_GIF=OFF -DENABLE_JPEG=OFF \
        -DENABLE_TIFF=OFF -DENABLE_WEBP=OFF -DENABLE_OPENJPEG=OFF
    cmake --build "$work/build-lept-$arch" --target install
    # shellcheck disable=SC2086
    cmake -S "$work/tesseract-5.5.3" -B "$work/build-tess-$arch" $common $target \
        -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_PREFIX_PATH="$prefix" -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TRAINING_TOOLS=OFF -DGRAPHICS_DISABLED=ON -DDISABLE_TIFF=ON -DDISABLE_ARCHIVE=ON -DDISABLE_CURL=ON \
        -DOPENMP_BUILD=OFF -DENABLE_NATIVE=OFF -DSW_BUILD=OFF -DINSTALL_CONFIGS=OFF -DENABLE_CCACHE=OFF \
        -DLEPT_TIFF_RESULT=1 -DLEPT_TIFF_RESULT__TRYRUN_OUTPUT=
    cmake --build "$work/build-tess-$arch" --target install
done

out="$dir/libtesseract.5.dylib"
lipo -create "$work/arm64/lib/libtesseract.dylib" "$work/x86_64/lib/libtesseract.dylib" -output "$out"
install_name_tool -id @loader_path/libtesseract.5.dylib "$out" 2>/dev/null
codesign --force --sign - "$out"
cp "$work/tesseract-5.5.3/LICENSE" "$dir/LICENSE-tesseract"
cp "$work/leptonica-1.87.0/leptonica-license.txt" "$dir/LICENSE-leptonica"

others=$(otool -L "$out" | awk '/^[[:space:]]/ {print $1}' | grep -v -E '^(/usr/lib/|/System/|@loader_path/libtesseract)' || true)
[ -z "$others" ] || { echo "needs libraries outside macOS:" >&2; echo "$others" >&2; exit 1; }
echo "built: $out ($(lipo -archs "$out"); lowest macOS $(otool -l "$out" | awk '/minos/ {print $2; exit}'))"
