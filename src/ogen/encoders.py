"""CIFAR-native feature encoder for Stage-2 matching.

Branches (both CIFAR stem: 3x3 conv, stride 1, no maxpool; native 32x32 input; CIFAR mean/std):
  simclr: SimCLR ResNet-34 trained on CIFAR-10 (edadaltocg/resnet34_simclr_cifar10, 500 ep, 90.0% linear acc.)
  cls:    supervised ResNet-18 classifier on CIFAR-10 (edadaltocg/resnet18_cifar10, 95.0% test acc.)
  -> self-supervised + label-trained, as the paper's ResNet-MAE + ConvNeXt pair. Outputs mirror the paper's ConvNeXt
`get_activations` (third_party/priming/.../convnext.py): per stage, features normalised over channels at each
position; full maps for stages 1-3, mean/std for all stages, global mean/std -> 13 outputs.
Same interface as posttrain.ConvNextEncoder (family "simclr"): input [0,1] BCHW.
"""
import pathlib
import timm
import torch
import torch.nn as nn

from .paths import OUT
SIMCLR34 = OUT / "downloads/encoders/resnet34_simclr_cifar10/pytorch_model.bin"
CLS18 = OUT / "downloads/encoders/resnet18_cifar10/pytorch_model.bin"
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
FCLS = OUT / "downloads/encoders/fmnist_fcls"   # trained by scripts/train_fmnist_clf.py (Fashion-MNIST, 32px padded)


def _branch(b):
    """-> (arch, weights path, mean, std). fcls: Fashion-MNIST classifier, grayscale repeated to 3 channels."""
    if b == "simclr":
        return "resnet34", SIMCLR34, MEAN, STD
    if b == "cls":
        return "resnet18", CLS18, MEAN, STD
    if b in ("fcls", "mcls"):   # Fashion-MNIST / MNIST classifier (scripts/train_fmnist_clf.py)
        import json
        d = FCLS if b == "fcls" else OUT / "downloads/encoders/mnist_mcls"
        m = json.loads((d / "meta.json").read_text())
        return m["arch"], d / "pytorch_model.bin", (m["mean"],) * 3, (m["std"],) * 3
    raise ValueError(b)


def _normalize(y):  # as convnext.get_activations: zero-mean/unit-std over channels at each position
    return (y - y.mean(1, keepdim=True)) / (y.std(1, keepdim=True, correction=0) + 1e-3)


def _safe_std(y, dim):
    return (y.var(dim, correction=0) + 1e-6).sqrt()


def _cifar_resnet(arch, path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}   # classifier head unused (features only)
    m = timm.create_model(arch, pretrained=False, num_classes=0)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    m.load_state_dict(sd, strict=True)
    return m


class SimCLRResNetEncoder(nn.Module):
    """branches=("simclr",) or ("simclr", "cls"); each branch gives 13 outputs named <branch>__<output>."""

    def __init__(self, device, microbatch=256, branches=("simclr",)):
        super().__init__()
        self.family_names = tuple(branches)
        cfg = {b: _branch(b) for b in branches}
        self.nets = nn.ModuleDict({b: _cifar_resnet(cfg[b][0], cfg[b][1]) for b in branches}).to(device).eval().requires_grad_(False)
        self.microbatch = microbatch
        self.norm = {b: (torch.tensor(cfg[b][2], device=device).view(1, 3, 1, 1),
                         torch.tensor(cfg[b][3], device=device).view(1, 3, 1, 1)) for b in branches}

    def encode_family(self, images, family="simclr"):
        with torch.autocast(images.device.type, enabled=False):
            m = self.nets[family]
            images = images.float()
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            mean, std = self.norm[family]
            x = (images - mean) / std
            x = m.act1(m.bn1(m.conv1(x)))
            out = {}
            for i, layer in enumerate((m.layer1, m.layer2, m.layer3, m.layer4)):
                x = layer(x)
                xn = _normalize(x)
                if i > 0:
                    out[f"stage_{i}"] = xn.flatten(2).transpose(1, 2)          # b, hw, c
                out[f"stage_{i}_mean"] = xn.mean((2, 3))[:, None]
                out[f"stage_{i}_std"] = _safe_std(xn.flatten(2), 2)[:, None]
            out["global_mean"] = x.mean((2, 3))[:, None]
            out["global_std"] = _safe_std(_normalize(x).flatten(2), 2)[:, None]
            return {f"{family}__{k}": v.flatten(1) for k, v in out.items()}

    def forward(self, images):
        out = {}
        for fam in self.family_names:
            parts = [self.encode_family(p, fam) for p in images.split(self.microbatch)]
            out.update({n: torch.cat([p[n] for p in parts]) for n in parts[0]})
        return out
