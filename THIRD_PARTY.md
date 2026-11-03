# Third-party code, weights and data

None of the items below is stored in this repo. The setup scripts download them from their original sources. You are responsible for following their terms. The status below was checked on 2026-11-03 and is a summary, not legal advice, so check the sources yourself.

## Code (`scripts/setup_third_party.sh` -> `third_party/priming/`)

| Item | Source | Licence |
|---|---|---|
| Sinkhorn matching loss, helpers (`losses/`, `utils/collectives.py`) | [JaeyeonKim01/priming](https://github.com/JaeyeonKim01/priming) | Apache-2.0 |
| ConvNeXt port (`models/encoders/full/vendor/convnext.py`) | same repo | that repo ships `LICENSE-W-Flow` (MIT, Jiaqi Han) next to it |

The script pins one commit and checks SHA-256 hashes of the files it uses.

`src/ogen/posttrain.py` follows the official Stage-2 recipe, and its `ConvNextEncoder` is adapted from the official `models/encoders/full/encoders.py`. Those parts are based on the Apache-2.0 Priming code, Copyright its authors, and were modified for this project (input resolution, encoder stack, datasets).

## Weights (`scripts/setup_weights.sh` -> `$OGEN_OUT/downloads/encoders/`)

| Item | Source | Licence |
|---|---|---|
| ConvNeXt-V2-Base, ImageNet-22k | [facebook/convnextv2-base-22k-224](https://huggingface.co/facebook/convnextv2-base-22k-224) | **Sources disagree; treat as non-commercial.** The Hugging Face card says Apache-2.0, while the [ConvNeXt-V2 repo](https://github.com/facebookresearch/ConvNeXt-V2) says its ImageNet pre-trained and fine-tuned models are CC-BY-NC |
| ResNet-18, CIFAR-10 classifier | [edadaltocg/resnet18_cifar10](https://huggingface.co/edadaltocg/resnet18_cifar10) | MIT (Hugging Face card) |
| ResNet-34 SimCLR, optional (`--encoder simclr`, not used for the reported results) | [edadaltocg/resnet34_simclr_cifar10](https://huggingface.co/edadaltocg/resnet34_simclr_cifar10) | MIT (Hugging Face card); not downloaded by default |
| Inception v3 for FID | downloaded by [torch-fidelity](https://github.com/toshas/torch-fidelity) on first use | see torch-fidelity |

No checkpoints trained in this project are published.

## Datasets (`scripts/setup_data.py` -> `$OGEN_DATA/`)

Downloaded through `torchvision` from the sources it uses. See the terms of [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).
