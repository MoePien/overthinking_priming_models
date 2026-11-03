"""One-step evaluation: Y_hat = D(X), X ~ N(0,I), exactly one network evaluation.

Metrics vs CIFAR-10 train (50k): FID-50k, KID (100 subsets x 1000), precision/recall (Kynkaanniemi k=3,
10k vs 10k), all on torch-fidelity Inception-v3 pool3 (2048) features. Plus:
  - output stats: across-sample per-pixel std, mean pairwise L2, mean distance to nearest train image
  - diagnostic D(Z) on held-out test pairs (independent and OT-paired, chunk 128): MSE and FID-10k vs test
  - fixed-seed grid (same seeds for every model)
Run: python -m src.ogen.evaluate --ckpt <run>/ema_0200000.pt --out <run>/eval_0200000.json
     python -m src.ogen.evaluate --empirical  (closed-form empirical-optimal independent denoiser)
"""
import argparse, json, pathlib, time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vu
from scipy import linalg

from .data import cifar_uint8, cifar_labels, to_float, to_uint8, dataset_uint8, dataset_labels
from .unet import UNet
from .coupling import interp, ot_perm, ot_perm_within, S

from .paths import OUT
REF = OUT / "results" / "ref_feats"


def inception():
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    return FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).cuda().eval()


@torch.no_grad()
def feats(fe, x_u8, bs=200):
    rgb = lambda x: x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x   # grayscale -> RGB for Inception
    return torch.cat([fe(rgb(x_u8[i:i + bs].cuda()))[0].double().cpu() for i in range(0, len(x_u8), bs)]).numpy()


def ref_feats(fe, split, dataset="cifar10"):
    REF.mkdir(parents=True, exist_ok=True)
    p = REF / (f"{split}.npy" if dataset == "cifar10" else f"{dataset}_{split}.npy")
    if not p.exists():
        np.save(p, feats(fe, dataset_uint8(dataset, split)))
    return np.load(p)


@torch.no_grad()
def feval_feats(x_u8, bs=1000, ds="fmnist"):
    """Fashion-MNIST only: pooled features of the held-out `feval` classifier (never used in training)."""
    import json, timm
    import torch.nn as nn
    d = OUT / ("downloads/encoders/fmnist_feval" if ds == "fmnist" else "downloads/encoders/mnist_meval")
    meta = json.loads((d / "meta.json").read_text())
    m = timm.create_model(meta["arch"], pretrained=False, num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = nn.Identity()
    m.load_state_dict(torch.load(d / "pytorch_model.bin", map_location="cpu")); m = m.cuda().eval()
    out = []
    for i in range(0, len(x_u8), bs):
        x = (x_u8[i:i + bs].cuda().float() / 255 - meta["mean"]) / meta["std"]
        out.append(m.forward_head(m.forward_features(x.repeat(1, 3, 1, 1)), pre_logits=True).double().cpu())
    return torch.cat(out).numpy()


def fid(a, b):
    m1, m2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cs, _ = linalg.sqrtm(s1 @ s2, disp=False)
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs.real))


def kid(a, b, n_sub=100, sz=1000, seed=0):
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    vals = []
    for _ in range(n_sub):
        x = a[rng.choice(len(a), sz, replace=False)]
        y = b[rng.choice(len(b), sz, replace=False)]
        k = lambda u, v: (u @ v.T / d + 1) ** 3
        kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
        m = sz
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def prec_rec(real, fake, k=3, n=10000, seed=0):
    """Improved precision/recall (Kynkaanniemi et al. 2019) on GPU."""
    rng = np.random.default_rng(seed)
    n = min(n, len(real), len(fake))
    r = torch.from_numpy(real[rng.choice(len(real), n, replace=False)]).float().cuda()
    f = torch.from_numpy(fake[rng.choice(len(fake), n, replace=False)]).float().cuda()

    def radii(x):
        d = torch.cdist(x, x)
        return d.kthvalue(k + 1, dim=1).values

    rr, rf = radii(r), radii(f)
    prec = (torch.cdist(f, r) <= rr[None]).any(1).float().mean().item()
    rec = (torch.cdist(r, f) <= rf[None]).any(1).float().mean().item()
    return prec, rec


class EmpiricalDenoiser:
    """Exact minimiser of E||D(Z)-Y||^2 over ALL functions, for Y ~ uniform(train set), X indep.
    D*(z) = sum_i softmax_i(-||z - (1-S) y_i||^2 / (2 S^2)) y_i."""

    def __init__(self, ytrain):
        self.y = ytrain.cuda().flatten(1)          # N x D
        self.yn = self.y.pow(2).sum(1)

    @torch.no_grad()
    def __call__(self, z):
        zf = z.flatten(1)
        logits = ((1 - S) * zf @ self.y.T - 0.5 * (1 - S) ** 2 * self.yn[None]) / S ** 2
        w = logits.softmax(1)
        self.last_maxw = w.max(1).values
        return (w @ self.y).view_as(z)


@torch.no_grad()
def gen(net, n, seed=0, bs=500, num_classes=0, ch=3):
    """Sample i gets class i % num_classes (class-balanced, as the paper's 50 per ImageNet class)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(0, n, bs):
        x = torch.randn((min(bs, n - i), ch, 32, 32), generator=g).cuda()
        if num_classes:
            c = torch.arange(i, i + len(x), device="cuda") % num_classes
            out.append(to_uint8(net(x, c).float()).cpu())
        else:
            out.append(to_uint8(net(x).float()).cpu())
    return torch.cat(out)


@torch.no_grad()
def gen_fm(net, n, steps, seed=0, bs=500, num_classes=0, ch=3):
    """Flow-matching net (x-prediction): Euler ODE from noise (t=0) to data (t=1), v = (D(x_t,t) - x_t)/(1-t).
    Same noise/class assignment as gen(); the last Euler step lands exactly on D(x_t, t)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(0, n, bs):
        x = torch.randn((min(bs, n - i), ch, 32, 32), generator=g).cuda()
        c = torch.arange(i, i + len(x), device="cuda") % num_classes if num_classes else None
        for k in range(steps):
            t = torch.full((len(x),), k / steps, device="cuda")
            d = net(x, c, t).float()
            x = x + (d - x) / (1 - k / steps) / steps
        out.append(to_uint8(x).cpu())
    return torch.cat(out)


@torch.no_grad()
def nn_dist(x_u8, train_u8, n=2000):
    """Mean L2 (in [-1,1] units) from n samples to their nearest train image."""
    x = to_float(x_u8[:n]).cuda().flatten(1)
    t = to_float(train_u8).cuda().flatten(1)
    best = torch.full((len(x),), float("inf"), device="cuda")
    for i in range(0, len(t), 5000):
        best = torch.minimum(best, torch.cdist(x, t[i:i + 5000]).min(1).values)
    return best.mean().item()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--empirical", action="store_true")
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fm_steps", type=int, default=25, help="flow-matching Stage-1 runs: Euler steps for the FID")
    a = ap.parse_args()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    ds, rcfg = "cifar10", {}
    if not a.empirical:
        ck = torch.load(a.ckpt, map_location="cuda")
        ds = "fmnist" if ck["ema"]["inp.weight"].shape[1] == 1 else "cifar10"
        cfg = pathlib.Path(a.ckpt).parent / "config.json"      # run config names the dataset (fmnist vs mnist)
        if cfg.exists():
            rcfg = json.loads(cfg.read_text())
            ds = rcfg.get("dataset", ds)
    CH = 1 if ds in ("fmnist", "mnist") else 3
    train_u8, test_u8 = dataset_uint8(ds, "train"), dataset_uint8(ds, "test")
    if a.empirical:
        net = EmpiricalDenoiser(to_float(train_u8))
        bs = 100
    else:
        K = ck["ema"]["cemb.0.weight"].shape[0] if "cemb.0.weight" in ck["ema"] else 0
        tc = "temb.0.weight" in ck["ema"]
        net = UNet(num_classes=K, in_ch=CH, time_cond=tc).cuda().eval()
        net.load_state_dict(ck["ema"])
        if tc:  # Stage 2 from a flow-matching init: its frozen t; flow-matching Stage 1: diagnostics at t = 1 - noise
            net.fixed_t = rcfg["fix_t"] if rcfg.get("fix_t") is not None else 1 - rcfg.get("noise", 0.8)
        bs = 500
    K = 0 if a.empirical else K
    fm_stage1 = rcfg.get("objective") == "fm"

    def run(x, c=None):
        if not K:
            return torch.cat([net(x[i:i + bs]) for i in range(0, len(x), bs)])
        return torch.cat([net(x[i:i + bs], c[i:i + bs]) for i in range(0, len(x), bs)])
    fe = inception()
    rtr, rte = ref_feats(fe, "train", ds), ref_feats(fe, "test", ds)

    res = {"ckpt": a.ckpt, "empirical": a.empirical, "n": a.n, "num_classes": K}
    with torch.autocast("cuda", torch.bfloat16, enabled=not a.empirical):
        if fm_stage1:   # headline = multi-step flow-matching sample quality (Euler, --fm_steps NFE)
            x_u8 = gen_fm(net, a.n, a.fm_steps, seed=0, bs=bs, num_classes=K, ch=CH)
            res["fm_steps"] = a.fm_steps
        else:
            x_u8 = gen(net, a.n, seed=0, bs=bs, num_classes=K, ch=CH)
    if a.empirical:
        net(torch.randn((1000, 3, 32, 32), generator=torch.Generator().manual_seed(0)).cuda())
        res["emp_maxw_on_X_mean"] = net.last_maxw.mean().item()
    f = feats(fe, x_u8)
    res["fid"] = fid(f, rtr)
    res["dataset"] = ds
    evdir = {"fmnist": "fmnist_feval", "mnist": "mnist_meval"}.get(ds)
    if evdir and (OUT / f"downloads/encoders/{evdir}/meta.json").exists():
        fp = REF / f"{ds}_train_feval.npy"     # Frechet distance in a held-out classifier's features
        if not fp.exists():
            np.save(fp, feval_feats(train_u8, ds=ds))
        res["fd_feval"] = fid(feval_feats(x_u8, ds=ds), np.load(fp))
    res["kid"], res["kid_std"] = kid(f, rtr)
    res["precision"], res["recall"] = prec_rec(rtr, f)
    xf = to_float(x_u8[:5000]).cuda()
    res["gen_pix_std"] = xf.std(0).mean().item()
    res["data_pix_std"] = to_float(train_u8[:5000]).std(0).mean().item()
    res["gen_pdist"] = torch.pdist(xf[:2000].flatten(1)).mean().item()
    res["data_pdist"] = torch.pdist(to_float(train_u8[:2000]).cuda().flatten(1)).mean().item()
    res["gen_nn_train_l2"] = nn_dist(x_u8, train_u8)
    res["test_nn_train_l2"] = nn_dist(test_u8, train_u8)
    # fixed-seed grid (first 64 of seed 0 -> identical noise across models)
    out = pathlib.Path(a.out)
    vu.save_image(to_float(x_u8[:100]).add(1).div(2), out.with_suffix(".grid.png"), nrow=10)

    # diagnostic: D(Z) on held-out test pairs (same construction as train.py val, but all 10k)
    g = torch.Generator().manual_seed(1234)
    yt = to_float(test_u8).cuda()
    xv = torch.randn(yt.shape, generator=g).cuda()
    lt = dataset_labels(ds, "test").cuda() if K else None
    if K:  # within-class OT keeps labels aligned: yt[perm] has labels lt
        perm = torch.cat([ot_perm_within(xv[i:i + 128], yt[i:i + 128], lt[i:i + 128])[0] + i
                          for i in range(0, len(yt), 128)])
    else:
        perm = torch.cat([ot_perm(xv[i:i + 128], yt[i:i + 128])[0] + i for i in range(0, len(yt), 128)])
    for k, y in (("ind", yt), ("ot", yt[perm])):
        with torch.autocast("cuda", torch.bfloat16, enabled=not a.empirical):
            d = run(interp(xv, y), lt).float()
        res[f"dz_mse_{k}"] = (d - y).pow(2).mean().item()
        res[f"dz_fid10k_test_{k}"] = fid(feats(fe, to_uint8(d).cpu()), rte)
        res[f"dz_pix_std_{k}"] = d.std(0).mean().item()
        if k == "ind":
            vu.save_image(torch.cat([y[:10], interp(xv, y)[:10].clamp(-1, 1), d[:10]]).add(1).div(2),
                          out.with_suffix(".dz.png"), nrow=10)
    # input-shift diagnostic (NOT the one-step metric): D(c*X) with c scaling X onto the norm shell of Z
    xs = torch.randn((10000, CH, 32, 32), generator=torch.Generator().manual_seed(0)).cuda()
    for c in (0.8, 0.81):
        with torch.autocast("cuda", torch.bfloat16, enabled=not a.empirical):
            d = run(c * xs, torch.arange(len(xs), device="cuda") % K if K else None).float()
        res[f"dcx{c}_fid10k_train"] = fid(feats(fe, to_uint8(d).cpu()), rtr)
        res[f"dcx{c}_pix_std"] = d.std(0).mean().item()
    # input-shift stats: per-pixel mean/std of Z vs X
    z = interp(xv, yt)
    res["z_mean"], res["z_std"] = z.mean().item(), z.std().item()
    res["z_norm"], res["x_norm"] = z.flatten(1).norm(dim=1).mean().item(), xv.flatten(1).norm(dim=1).mean().item()
    res["eval_sec"] = time.time() - t0
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
