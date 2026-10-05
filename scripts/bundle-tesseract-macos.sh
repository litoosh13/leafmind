#!/bin/sh
# Copies Homebrew's Tesseract library and every non-system library it needs into one folder, so an app can
# ship them: references are rewritten to @loader_path (the libraries find each other in that folder) and each
# file is signed again ad hoc (required on Apple silicon after changing it; an app signs them with its own
# identity when it is signed). Needs `brew install tesseract`. Point leafmind-ocr at <dir>/libtesseract.5.dylib.
# The libraries keep the lowest macOS version Homebrew built them for (printed at the end); an app for older
# macOS versions has to build Tesseract itself with MACOSX_DEPLOYMENT_TARGET set.
# Usage: scripts/bundle-tesseract-macos.sh <dir>
set -eu
dir=${1:?usage: $0 <dir>}
prefix=$(brew --prefix)
mkdir -p "$dir"

# References to libraries that are not part of macOS: paths under the Homebrew prefix and @rpath names.
refs() { otool -L "$1" | tail -n +2 | awk '{print $1}' | grep -E "^($prefix/|@rpath/)" || true; }
# The file behind a reference (Homebrew links every library into its lib folder).
source_of() { case "$1" in @rpath/*) echo "$prefix/lib/${1#@rpath/}" ;; *) echo "$1" ;; esac; }

queue="$prefix/opt/tesseract/lib/libtesseract.5.dylib"
done_list=""
while [ -n "$queue" ]; do
    lib=${queue%%
*}
    [ "$lib" = "$queue" ] && queue="" || queue=${queue#*
}
    name=$(basename "$lib")
    case " $done_list " in *" $name "*) continue ;; esac
    done_list="$done_list $name"
    cp -L "$lib" "$dir/$name"
    chmod u+w "$dir/$name"
    for ref in $(refs "$lib"); do
        queue="$queue${queue:+
}$(source_of "$ref")"
    done
done

for name in $done_list; do
    file="$dir/$name"
    install_name_tool -id "@loader_path/$name" "$file" 2>/dev/null
    for ref in $(refs "$file"); do
        install_name_tool -change "$ref" "@loader_path/$(basename "$ref")" "$file" 2>/dev/null
    done
    codesign --force --sign - "$file" 2>/dev/null
done

echo "copied:$done_list"
left=$(for name in $done_list; do refs "$dir/$name"; done)
[ -z "$left" ] || { echo "still pointing into $prefix:" >&2; echo "$left" >&2; exit 1; }
minos=$(otool -l "$dir/libtesseract.5.dylib" | awk '/minos/ {print $2; exit}')
echo "lowest macOS version: $minos"
