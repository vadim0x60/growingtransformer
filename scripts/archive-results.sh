#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 SOURCE_RUN_DIRECTORY DESTINATION_RESULTS_DIRECTORY" >&2
    exit 2
fi

SOURCE=$(realpath "$1")
DESTINATION=$(realpath -m "$2")

test -d "$SOURCE"
test ! -e "$DESTINATION"
command -v zstd >/dev/null

if find "$SOURCE" -type l -print -quit | grep -q .; then
    echo "Refusing to archive symlinks; result archives must be self-contained" >&2
    exit 1
fi

PARENT=$(dirname "$DESTINATION")
mkdir -p "$PARENT"
TEMP=$(mktemp -d "$PARENT/.archive-results.XXXXXX")
trap 'rm -rf "$TEMP"' EXIT
OUTPUT="$TEMP/output"
EXTRACTED="$TEMP/extracted"
FILE_LIST="$TEMP/files.null"
mkdir "$OUTPUT" "$EXTRACTED"

find "$SOURCE" -type f ! -name '*.tmp' -printf '%P\0' | sort -z > "$FILE_LIST"
test -s "$FILE_LIST"

while IFS= read -r -d '' relative; do
    sha256sum "$SOURCE/$relative" | awk -v path="$relative" '{print $1 "  " path}'
done < "$FILE_LIST" > "$OUTPUT/SHA256SUMS"

tar --create \
    --file "$OUTPUT/artifacts.tar.zst" \
    --directory "$SOURCE" \
    --no-recursion --sort=name \
    --format=pax --pax-option=delete=atime,delete=ctime \
    --mtime='@0' --owner=0 --group=0 --numeric-owner \
    --mode='u+rwX,go+rX,go-w' \
    --use-compress-program='zstd -10 -T1 --no-progress' \
    --null --files-from "$FILE_LIST"

sha256sum "$OUTPUT/artifacts.tar.zst" |
    awk '{print $1 "  artifacts.tar.zst"}' > "$OUTPUT/artifacts.tar.zst.sha256"

zstd --quiet --test "$OUTPUT/artifacts.tar.zst"
tar --extract --file "$OUTPUT/artifacts.tar.zst" --directory "$EXTRACTED" \
    --use-compress-program=zstd
(
    cd "$EXTRACTED"
    sha256sum --check "$OUTPUT/SHA256SUMS"
) >/dev/null

mv "$OUTPUT" "$DESTINATION"
printf 'Archived %s files (%s bytes) to %s\n' \
    "$(wc -l < "$DESTINATION/SHA256SUMS")" \
    "$(stat -c '%s' "$DESTINATION/artifacts.tar.zst")" \
    "$DESTINATION"
