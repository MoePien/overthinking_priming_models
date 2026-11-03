#!/usr/bin/env bash
# Fetches the parts of the official Priming code this project imports into third_party/priming/.
# Nothing is redistributed by this repo. Licences: see THIRD_PARTY.md.
set -euo pipefail
cd "$(dirname "$0")/.."
URL=https://github.com/JaeyeonKim01/priming
PIN=bf68e2a6a14000603eb270cc93830514fc6bdd14
DEST=third_party/priming
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

git clone --quiet "$URL" "$TMP/src"
git -C "$TMP/src" checkout --quiet "$PIN"

(cd "$TMP/src" && sha256sum --check --quiet) <<'EOF'
4ca53f8f9f89cc0bf45f189653eba02d95a93461ec7176f36e7811bb74c56bbe  losses/discrepancy.py
7c9ba7b27f17430e6f7ffaee3dcf68aec6ab59a96c69a2a5dbf1ce010e570901  losses/matching.py
05ae4860d7bbd2e9b4c8a7fd76bf93ea8eaaac0c017c036e94de39263e6f6ba5  utils/collectives.py
f4077ce0a8f675517e7a301046a03fa74ccdb5a8ce7e72b5e3cbfab9426ef63f  models/encoders/full/vendor/convnext.py
ee12cc1a4b06a05a337d9fb70b72867afaed7072d6ead0dfd3f64fd2249e3c12  models/encoders/full/LICENSE-W-Flow
EOF

for f in losses/discrepancy.py losses/matching.py utils/collectives.py \
         models/encoders/full/vendor/convnext.py models/encoders/full/LICENSE-W-Flow LICENSE; do
  mkdir -p "$DEST/$(dirname "$f")"
  cp "$TMP/src/$f" "$DEST/$f"
done
for d in losses utils models models/encoders models/encoders/full models/encoders/full/vendor; do
  : > "$DEST/$d/__init__.py"
done
echo "OK: $DEST filled from $URL @ $PIN"
