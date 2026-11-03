"""Download CIFAR-10, MNIST and Fashion-MNIST into $OGEN_DATA (default ./data) via torchvision. Terms: see THIRD_PARTY.md."""
import pathlib, sys

from torchvision import datasets

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.ogen.paths import DATA  # noqa: E402

for name, cls in (("cifar10", datasets.CIFAR10), ("mnist", datasets.MNIST), ("fmnist", datasets.FashionMNIST)):
    for train in (True, False):
        cls(str(DATA / name), train=train, download=True)
    print("OK:", DATA / name)
