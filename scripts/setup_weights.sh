#!/usr/bin/env bash
# Downloads the Stage-2 feature networks from Hugging Face at pinned revisions and checks their hashes.
# Check the licences first: see THIRD_PARTY.md. The ConvNeXt-V2 weights should be treated as non-commercial.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OGEN_OUT:-$PWD/outputs}/downloads/encoders

get() {  # local name, repo, revision, sha256
  mkdir -p "$OUT/$1"
  curl -fL --retry 3 -o "$OUT/$1/pytorch_model.bin" "https://huggingface.co/$2/resolve/$3/pytorch_model.bin"
  echo "$4  $OUT/$1/pytorch_model.bin" | sha256sum --check --quiet
  echo "OK: $1"
}

get convnextv2-base-22k-224 facebook/convnextv2-base-22k-224 758ff0922dc09136abb55774e7f8b1e1bd0dc344 \
    175a54bbcf71a1a34b26143db4f0f24ddea99ef30981d596860d43b4b76031c1
get resnet18_cifar10 edadaltocg/resnet18_cifar10 ccacbae365101b9bfa45e1f17f4acac30c2601b2 \
    f656242526b567dbd46857276acbcfcb4ee74a7f8a7ae5e56557b7e8fa5d914a
