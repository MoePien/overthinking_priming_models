"""CIFAR-10 from the local python pickles, as uint8 tensors / [-1,1] floats."""
import pickle
import numpy as np
import torch

from .paths import DATA

CIFAR_DIR = f"{DATA}/cifar10/cifar-10-batches-py"


def _load(names, labels=False):
    xs, ls = [], []
    for n in names:
        with open(f"{CIFAR_DIR}/{n}", "rb") as f:
            d = pickle.load(f, encoding="bytes")
        xs.append(d[b"data"].reshape(-1, 3, 32, 32))
        ls.append(np.asarray(d[b"labels"]))
    if labels:
        return torch.from_numpy(np.concatenate(ls)).long()
    return torch.from_numpy(np.concatenate(xs))  # uint8 N,3,32,32


def cifar_uint8(split="train"):
    if split == "train":
        return _load([f"data_batch_{i}" for i in range(1, 6)])
    return _load(["test_batch"])


def cifar_labels(split="train"):
    if split == "train":
        return _load([f"data_batch_{i}" for i in range(1, 6)], labels=True)
    return _load(["test_batch"], labels=True)


FMNIST_DIR = f"{DATA}/fmnist/FashionMNIST/raw"
MNIST_DIR = f"{DATA}/mnist/MNIST/raw"
IDX_DIRS = {"fmnist": FMNIST_DIR, "mnist": MNIST_DIR}
CHANNELS = {"cifar10": 3, "fmnist": 1, "mnist": 1}
HFLIP = {"cifar10": True, "fmnist": True, "mnist": False}   # no mirror augmentation for digits


def _idx(path):
    with open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def dataset_uint8(name="cifar10", split="train"):
    """uint8 (N, C, 32, 32). fmnist/mnist: 28x28 grayscale zero-padded by 2 px to 32x32 (background is 0)."""
    if name == "cifar10":
        return cifar_uint8(split)
    if name in IDX_DIRS:
        x = _idx(f"{IDX_DIRS[name]}/{'train' if split == 'train' else 't10k'}-images-idx3-ubyte")
        return torch.from_numpy(np.pad(x, ((0, 0), (2, 2), (2, 2)))[:, None].copy())
    raise ValueError(name)


def dataset_labels(name="cifar10", split="train"):
    if name == "cifar10":
        return cifar_labels(split)
    if name in IDX_DIRS:
        return torch.from_numpy(_idx(f"{IDX_DIRS[name]}/{'train' if split == 'train' else 't10k'}-labels-idx1-ubyte")
                                .astype(np.int64))
    raise ValueError(name)


def to_float(x_u8):
    return x_u8.float().div(127.5).sub(1.0)


def to_uint8(x):
    return x.clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)


class GPUSampler:
    """Whole dataset on GPU; random batches with optional horizontal flip."""

    def __init__(self, x_u8, device, hflip=True, seed=0, labels=None):
        self.x = to_float(x_u8).to(device)
        self.hflip = hflip
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.labels = None if labels is None else labels.to(device)
        if labels is not None:
            self.by_class = [torch.nonzero(self.labels == k)[:, 0] for k in range(int(labels.max()) + 1)]

    def __call__(self, b):
        """b random images; their labels (if any) in self.last_labels."""
        idx = torch.randint(0, self.x.shape[0], (b,), device=self.x.device, generator=self.g)
        if self.labels is not None:
            self.last_labels = self.labels[idx]
        return self._get(idx)

    def per_class(self, classes, n):
        """n random images of each class in `classes` (concatenated in that order)."""
        idx = torch.cat([self.by_class[int(k)][torch.randint(0, len(self.by_class[int(k)]), (n,),
                         device=self.x.device, generator=self.g)] for k in classes])
        return self._get(idx)

    def _get(self, idx):
        y = self.x[idx]
        if self.hflip:
            f = torch.rand(len(idx), device=y.device, generator=self.g) < 0.5
            y = torch.where(f[:, None, None, None], y.flip(-1), y)
        return y
