# Overthinking priming models

Experiments on [Priming Models](https://arxiv.org/abs/2610.20331) ("Are We Overthinking One-Step Generation?", Park, Kim, Du, Chen, Han) on CIFAR-10, plus the flow-matching start on MNIST and Fashion-MNIST. All generation is class-conditional, with one network call per sample.

This is unofficial research code, not a replication of the paper. It is not affiliated with the paper's authors. The repo holds code, exact commands and metric logs. It has no checkpoints, datasets or third-party code; see [Setup](#setup) to fetch those.

## The idea

**Priming Models.** Stage 1 trains a network to reconstruct clean images from corrupted ones. Stage 2 feeds it pure Gaussian noise and post-trains it as a one-step generator with a distributional loss in a frozen feature space.

The paper's Stage 1 is flexible. Its general form draws the noise level `s` from a distribution. A fixed `s = 0.8` is its simplest instantiation and its ImageNet default, not the whole method. The paper also compares other fixed levels, uses a logit-normal noise schedule for text-to-image, tries masking and RAE distillation as alternatives to denoising, and adds a perceptual loss to Stage 1 on ImageNet. We study only part of this: pixel MSE, either at a fixed `s` or over all times.

**Stage 1 as flow matching (our reading, not the paper's framing).** A reconstruction loss at fixed `s` is flow matching with x-prediction at one fixed time. With `Z_t = (1-t)X + tY` and `t = 1-s` (`Y` data, `X` noise), the velocity `v = (D - Z_t)/(1-t)` gives

```
||v - (Y - X)||^2 = ||D - Y||^2 / (1-t)^2
```

This suggests two changes to Stage 1:

- **Coupling:** flow matching allows any coupling, so we also pair noise and images by optimal transport (OT) within each class instead of independently.
- **Starting point:** a flow-matching network trained over all times can also start Stage 2. We freeze its time input at `t = 0.6`.

**Stage 2.** We keep the paper's OT post-training loss and use its official implementation. All Stage-1 variants above get the same Stage 2. With `a` the generated and `b` the real features of one class,

```
F(a, b) = D(a, b) - D(a, sg[a])
```

where `sg` is the stop-gradient, and `D` is an entropy-regularised OT cost on squared feature distances, computed per class and per feature output. The first term pulls generated samples towards real ones, and the second adds repulsion among generated samples. Optimiser, EMA and loss settings follow the paper's ImageNet recipe.

Our changes to Stage 2 for 32 px data:

- **Features:** ConvNeXt-V2-Base at 112 px plus a classifier trained on the dataset, weighted 1/2 each. The paper uses ConvNeXt-V2 and ResNet-MAE.
- **Batches:** each update uses 10 classes, each with 128 generated, 128 generated reference and 256 real images.
- **Budget:** we report 10k updates, and 25k for the CIFAR-10 flow-matching start. The paper uses 25k on ImageNet.

## Results

Stage-2 FID-50k against the training set, EMA weights, after 10k Stage-2 updates unless stated. **Every number is from a single seed**, so small gaps are not resolved.

**CIFAR-10** (test vs train reference: 3.15)

| s | independent | OT |
|---|---|---|
| 0.84 | 10.26 | 6.12 |
| 0.80 (paper's ImageNet default) | 7.88 | 5.42 |
| 0.75 | 6.27 | 4.81 |
| 0.70 | 5.23 | 4.04 |
| 0.65 | 4.48 | 3.70 |
| 0.55 | 3.57 | 3.33 |
| 0.45 | 3.25 | 3.03 |
| 0.35 | 3.12 | 3.05 |
| 0.25 | 3.25 | 3.24 |
| 0.15 | 4.14 | 4.03 |
| flow matching, t = 0.6 | 2.87 | 2.65 |
| flow matching, t = 0.6, 25k updates | **2.36** | **2.21** |

**MNIST and Fashion-MNIST, flow-matching start only** (`t = 0.6`, chosen on CIFAR-10 and not tuned for these datasets)

| | independent | OT | test vs train |
|---|---|---|---|
| MNIST | 1.16 | 0.99 | 1.02 |
| Fashion-MNIST | 1.67 | 1.45 | 1.42 |

What the CIFAR-10 table shows:

- OT is at or below independent pairing at 10 of 10 matched fixed noise levels. The gap is largest at high noise and closes at low noise.
- `s = 0.8` is far from the best fixed level. That is a different setting from the paper, whose noise-level comparison on ImageNet at 256 px favours 0.8 among the levels tested.
- The flow-matching start beats the best fixed level for both couplings, by 0.25 to 0.39 FID, but it also had 200k Stage-1 updates against 100k. `t = 0.6` sits near the best fixed level.

What it does not show:

- Which `s` is exactly best where gaps are small, for example around `s = 0.35` to `0.45`.
- Anything about ImageNet, or about the paper's other Stage-1 variants. Our Stage 1 has no perceptual loss, and our data and Stage-2 features differ from the paper's.

Flow-matching networks on their own (Euler, 25 steps, 200k updates, independent / OT): MNIST 0.68 / 0.60, Fashion-MNIST 3.58 / 3.30, CIFAR-10 8.21 / 7.86. On CIFAR-10, Stage-1 FID is not a good predictor of Stage 2: it gets worse as `s` falls while Stage 2 improves.

## Setup

Python 3.11. Everything downloaded goes to `third_party/` or to the directories set by two environment variables, and none of it is tracked by git:

```bash
export OGEN_DATA=/path/to/datasets      # default: ./data
export OGEN_OUT=/path/to/outputs        # default: ./outputs (runs, weights, caches)

pip install -r requirements.txt         # install the PyTorch build for your system first
bash scripts/setup_third_party.sh       # fetches the official Priming code at a pinned commit, checks hashes
bash scripts/setup_weights.sh           # fetches Stage-2 feature networks at pinned revisions, checks hashes
python scripts/setup_data.py            # fetches CIFAR-10, MNIST, Fashion-MNIST
```

Read [THIRD_PARTY.md](THIRD_PARTY.md) first. The downloads have their own licences, and some of them may restrict use.

## Running

`experiments/commands.sh` lists the exact arguments of every reported run. It would run all of them in sequence, so you will usually copy single lines. One example, CIFAR-10 at `s = 0.45` with OT pairing:

```bash
python -m src.ogen.train --bs 128 --steps 100000 --num_classes 10 --pairing epoch --coupling ot --noise 0.45 \
    --out $OGEN_OUT/runs/cifar10/s0450_ot/stage1
python -m src.ogen.posttrain --steps 10000 --keep_every 2500 --microbatch 64 --encoder convnext112+cls --gen_chunk 128 \
    --num_classes 10 --groups 10 --samples 128 --refs 256 \
    --init $OGEN_OUT/runs/cifar10/s0450_ot/stage1 --out $OGEN_OUT/runs/cifar10/s0450_ot/stage2
python -m src.ogen.evaluate --ckpt $OGEN_OUT/runs/cifar10/s0450_ot/stage2/ema_0010000.pt \
    --out $OGEN_OUT/runs/cifar10/s0450_ot/stage2/eval_0010000.json
```

The CIFAR-10 runs at `s = 0.8` were trained with an earlier revision of `src/ogen`. Its CIFAR-10 defaults are the same, but it is not this exact commit. Commands have not been re-run from a clean checkout beyond a short smoke test. Feature classifiers for MNIST and Fashion-MNIST are retrained by `scripts/train_fmnist_clf.py`, so a rerun gets different classifier weights from ours.

## Layout

| Path | Contents |
|---|---|
| `src/ogen/train.py` | Stage 1: fixed-noise denoiser or flow matching, independent or pool-OT pairing |
| `src/ogen/posttrain.py` | Stage 2: distributional post-training (recipe follows the official code) |
| `src/ogen/evaluate.py` | FID-50k, KID, precision/recall and diagnostics |
| `src/ogen/pairing.py`, `coupling.py` | OT pairing, exact within class on chunks of 2500-3000 images |
| `third_party/` | Empty until `scripts/setup_third_party.sh` has run |
| `experiments/commands.sh` | Exact commands of all reported runs |
| `results/summary.csv` | Stage-2 metrics at each evaluated budget |
| `results/logs/` | Per-run config, training log and evaluation metrics |

The logs keep hyperparameters and metrics only. Some `config.json` files show a larger `steps` value than reported here because those runs were continued further; the logs and tables stop at the reported budget.

## Licence

Our code is under the [MIT licence](LICENSE), except where [THIRD_PARTY.md](THIRD_PARTY.md) says otherwise. Third-party code, weights and datasets are not part of this repo and keep their own terms.
