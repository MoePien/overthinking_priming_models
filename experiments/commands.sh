#!/usr/bin/env bash
# Exact arguments of every reported run. Generated from the run configs. Stage 2 needs its Stage 1.
# Runs resume from last.pt. Learning rates are constant after warmup, so extending a run equals running it straight.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
export OGEN_OUT=${OGEN_OUT:-$PWD/outputs}
export PYTORCH_ALLOC_CONF=expandable_segments:True

# cifar10/s0840_ind
"$PYTHON" -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ind --noise 0.84 --out $OGEN_OUT/runs/cifar10/s0840_ind/stage1
"$PYTHON" -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --init $OGEN_OUT/runs/cifar10/s0840_ind/stage1 --out $OGEN_OUT/runs/cifar10/s0840_ind/stage2

# cifar10/s0800_ind
"$PYTHON" -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ind --noise 0.8 --out $OGEN_OUT/runs/cifar10/s0800_ind/stage1
"$PYTHON" -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --init $OGEN_OUT/runs/cifar10/s0800_ind/stage1 --out $OGEN_OUT/runs/cifar10/s0800_ind/stage2

# cifar10/s0750_ind
"$PYTHON" -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ind --noise 0.75 --out $OGEN_OUT/runs/cifar10/s0750_ind/stage1
"$PYTHON" -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --init $OGEN_OUT/runs/cifar10/s0750_ind/stage1 --out $OGEN_OUT/runs/cifar10/s0750_ind/stage2

# cifar10/s0700_ind
"$PYTHON" -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ind --noise 0.7 --out $OGEN_OUT/runs/cifar10/s0700_ind/stage1
"$PYTHON" -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --init $OGEN_OUT/runs/cifar10/s0700_ind/stage1 --out $OGEN_OUT/runs/cifar10/s0700_ind/stage2

# cifar10/s0650_ind
"$PYTHON" -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ind --noise 0.65 --out $OGEN_OUT/runs/cifar10/s0650_ind/stage1
"$PYTHON" -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --init $OGEN_OUT/runs/cifar10/s0650_ind/stage1 --out $OGEN_OUT/runs/cifar10/s0650_ind/stage2

# cifar10/s0550_ind
"$PYTHON" -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ind --noise 0.55 --out $OGEN_OUT/runs/cifar10/s0550_ind/stage1
"$PYTHON" -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --init $OGEN_OUT/runs/cifar10/s0550_ind/stage1 --out $OGEN_OUT/runs/cifar10/s0550_ind/stage2

# cifar10/s0450_ind
"$PYTHON" -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ind --noise 0.45 --out $OGEN_OUT/runs/cifar10/s0450_ind/stage1
"$PYTHON" -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --init $OGEN_OUT/runs/cifar10/s0450_ind/stage1 --out $OGEN_OUT/runs/cifar10/s0450_ind/stage2

# cifar10/s0350_ind
"$PYTHON" -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ind --noise 0.35 --out $OGEN_OUT/runs/cifar10/s0350_ind/stage1
"$PYTHON" -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --init $OGEN_OUT/runs/cifar10/s0350_ind/stage1 --out $OGEN_OUT/runs/cifar10/s0350_ind/stage2

# cifar10/s0250_ind
"$PYTHON" -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ind --noise 0.25 --out $OGEN_OUT/runs/cifar10/s0250_ind/stage1
"$PYTHON" -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --init $OGEN_OUT/runs/cifar10/s0250_ind/stage1 --out $OGEN_OUT/runs/cifar10/s0250_ind/stage2

# cifar10/s0150_ind
"$PYTHON" -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ind --noise 0.15 --out $OGEN_OUT/runs/cifar10/s0150_ind/stage1
"$PYTHON" -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --init $OGEN_OUT/runs/cifar10/s0150_ind/stage1 --out $OGEN_OUT/runs/cifar10/s0150_ind/stage2

# cifar10/fm_ind
"$PYTHON" -m src.ogen.train --objective fm --bs 128 --steps 200000 --keep_every 50000 --num_classes 10 --pairing epoch --coupling ind --noise 0.4 --out $OGEN_OUT/runs/cifar10/fm_ind/stage1
"$PYTHON" -m src.ogen.posttrain --steps 25000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --fix_t 0.6 --init $OGEN_OUT/runs/cifar10/fm_ind/stage1 --out $OGEN_OUT/runs/cifar10/fm_ind/stage2

# cifar10/s0840_ot
"$PYTHON" -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ot --noise 0.84 --out $OGEN_OUT/runs/cifar10/s0840_ot/stage1
"$PYTHON" -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --init $OGEN_OUT/runs/cifar10/s0840_ot/stage1 --out $OGEN_OUT/runs/cifar10/s0840_ot/stage2

# cifar10/s0800_ot
"$PYTHON" -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ot --noise 0.8 --out $OGEN_OUT/runs/cifar10/s0800_ot/stage1
"$PYTHON" -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --init $OGEN_OUT/runs/cifar10/s0800_ot/stage1 --out $OGEN_OUT/runs/cifar10/s0800_ot/stage2

# cifar10/s0750_ot
"$PYTHON" -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ot --noise 0.75 --out $OGEN_OUT/runs/cifar10/s0750_ot/stage1
"$PYTHON" -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --init $OGEN_OUT/runs/cifar10/s0750_ot/stage1 --out $OGEN_OUT/runs/cifar10/s0750_ot/stage2

# cifar10/s0700_ot
"$PYTHON" -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ot --noise 0.7 --out $OGEN_OUT/runs/cifar10/s0700_ot/stage1
"$PYTHON" -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --init $OGEN_OUT/runs/cifar10/s0700_ot/stage1 --out $OGEN_OUT/runs/cifar10/s0700_ot/stage2

# cifar10/s0650_ot
"$PYTHON" -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ot --noise 0.65 --out $OGEN_OUT/runs/cifar10/s0650_ot/stage1
"$PYTHON" -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --init $OGEN_OUT/runs/cifar10/s0650_ot/stage1 --out $OGEN_OUT/runs/cifar10/s0650_ot/stage2

# cifar10/s0550_ot
"$PYTHON" -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ot --noise 0.55 --out $OGEN_OUT/runs/cifar10/s0550_ot/stage1
"$PYTHON" -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --init $OGEN_OUT/runs/cifar10/s0550_ot/stage1 --out $OGEN_OUT/runs/cifar10/s0550_ot/stage2

# cifar10/s0450_ot
"$PYTHON" -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ot --noise 0.45 --out $OGEN_OUT/runs/cifar10/s0450_ot/stage1
"$PYTHON" -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --init $OGEN_OUT/runs/cifar10/s0450_ot/stage1 --out $OGEN_OUT/runs/cifar10/s0450_ot/stage2

# cifar10/s0350_ot
"$PYTHON" -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ot --noise 0.35 --out $OGEN_OUT/runs/cifar10/s0350_ot/stage1
"$PYTHON" -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --init $OGEN_OUT/runs/cifar10/s0350_ot/stage1 --out $OGEN_OUT/runs/cifar10/s0350_ot/stage2

# cifar10/s0250_ot
"$PYTHON" -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ot --noise 0.25 --out $OGEN_OUT/runs/cifar10/s0250_ot/stage1
"$PYTHON" -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --init $OGEN_OUT/runs/cifar10/s0250_ot/stage1 --out $OGEN_OUT/runs/cifar10/s0250_ot/stage2

# cifar10/s0150_ot
"$PYTHON" -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ot --noise 0.15 --out $OGEN_OUT/runs/cifar10/s0150_ot/stage1
"$PYTHON" -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --init $OGEN_OUT/runs/cifar10/s0150_ot/stage1 --out $OGEN_OUT/runs/cifar10/s0150_ot/stage2

# cifar10/fm_ot
"$PYTHON" -m src.ogen.train --objective fm --bs 128 --steps 200000 --keep_every 50000 --num_classes 10 --pairing epoch --coupling ot --noise 0.4 --out $OGEN_OUT/runs/cifar10/fm_ot/stage1
"$PYTHON" -m src.ogen.posttrain --steps 25000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 --num_classes 10 --groups 10 --samples 128 --refs 256 --fix_t 0.6 --init $OGEN_OUT/runs/cifar10/fm_ot/stage1 --out $OGEN_OUT/runs/cifar10/fm_ot/stage2

# FID-50k etc. for every kept checkpoint
for ck in "$OGEN_OUT"/runs/*/*/*/ema_*.pt; do
  js=${ck/ema_/eval_}; js=${js%.pt}.json
  [ -f "$js" ] || "$PYTHON" -m src.ogen.evaluate --ckpt "$ck" --out "$js"
done
