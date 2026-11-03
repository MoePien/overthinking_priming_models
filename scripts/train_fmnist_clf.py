"""Train a Fashion-MNIST / MNIST classifier with the CIFAR-style ResNet stem used by src/ogen/encoders.py.

  fcls  = resnet18, seed 0 -> Stage-2 feature branch (replaces the CIFAR classifier)
  feval = resnet34, seed 1 -> evaluation only (Frechet distance in its feature space; never used for training)
Input: 32x32 zero-padded grayscale repeated to 3 channels, normalised by the padded-train mean/std.
Saves <OUT>/downloads/encoders/fmnist_<name>/pytorch_model.bin (timm state dict incl. fc) + meta.json.
Run: python scripts/train_fmnist_clf.py fcls|feval
"""
import json, pathlib, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
import timm
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.ogen.data import dataset_uint8, dataset_labels, HFLIP

name = sys.argv[1]
arch, seed, DS = {"fcls": ("resnet18", 0, "fmnist"), "feval": ("resnet34", 1, "fmnist"),
                  "mcls": ("resnet18", 0, "mnist"), "meval": ("resnet34", 1, "mnist")}[name]
EPOCHS = 30 if DS == "fmnist" else 12
from src.ogen.paths import OUT as _OUT
OUT = _OUT / f"downloads/encoders/{DS}_{name}"
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(seed)
dev = "cuda"
xtr = dataset_uint8(DS, "train").float().div(255); ytr = dataset_labels(DS, "train")
xte = dataset_uint8(DS, "test").float().div(255); yte = dataset_labels(DS, "test")
mean, std = xtr.mean().item(), xtr.std().item()
xtr, xte = xtr.to(dev), xte.to(dev); ytr, yte = ytr.to(dev), yte.to(dev)

m = timm.create_model(arch, pretrained=False, num_classes=10)
m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
m.maxpool = nn.Identity()
m = m.to(dev).to(memory_format=torch.channels_last)
BS = 256
opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
steps = EPOCHS * (len(xtr) // BS)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=steps, pct_start=0.15)


def prep(x):
    return ((x - mean) / std).repeat(1, 3, 1, 1)


def augment(x):
    x = F.pad(x, (2, 2, 2, 2))
    i, j = torch.randint(0, 5, (2,))
    x = x[..., i:i + 32, j:j + 32]
    if not HFLIP[DS]:
        return x
    f = torch.rand(len(x), device=x.device) < 0.5
    return torch.where(f[:, None, None, None], x.flip(-1), x)


t0 = time.time()
for ep in range(EPOCHS):
    m.train()
    perm = torch.randperm(len(xtr), device=dev)
    for k in range(len(xtr) // BS):
        idx = perm[k * BS:(k + 1) * BS]
        with torch.autocast("cuda", torch.bfloat16):
            loss = F.cross_entropy(m(prep(augment(xtr[idx]))), ytr[idx], label_smoothing=0.1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    m.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        acc = torch.cat([m(prep(xte[i:i + 1000])).argmax(1) for i in range(0, len(xte), 1000)]).eq(yte).float().mean().item()
    print(f"epoch {ep + 1} loss {loss.item():.3f} test acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
torch.save(m.float().state_dict(), OUT / "pytorch_model.bin")
(OUT / "meta.json").write_text(json.dumps(dict(arch=arch, seed=seed, test_acc=acc, mean=mean, std=std, epochs=EPOCHS), indent=1))
print("saved", OUT, "acc", acc)
