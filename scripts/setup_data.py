"""Download CIFAR-10 into $OGEN_DATA (default ./data) via torchvision. Terms: see THIRD_PARTY.md."""
import pathlib, sys

from torchvision import datasets

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.ogen.paths import DATA  # noqa: E402

for train in (True, False):
    datasets.CIFAR10(str(DATA / "cifar10"), train=train, download=True)
print("OK:", DATA / "cifar10")
