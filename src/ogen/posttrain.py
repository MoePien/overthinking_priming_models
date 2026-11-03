"""Stage 2 (Priming Models distributional post-training) on CIFAR-10, unconditional.

Follows the official posttrain.py / losses/matching.py (vendored verbatim in third_party/priming):
  x = D(z) (live, with grad), x' = sg[D(z')] (independent reference), y = real; all in [0,1];
  loss = 5 * [OT_eps(phi(x), phi(y)) - OT_eps(phi(x), sg phi(x'))], featurewise over ConvNeXt-V2-B outputs,
  eps mixture 0.75*OT_0.001 + 0.25*OT_0.01 on cost ||a-b||^2 / (2 sigma_r^2), log-domain Sinkhorn (1000 it, tol 1e-6).
  AdamW(0.9,0.95) lr 1e-5, linear warmup 125, wd 0, clip 1; init = Stage-1 ONLINE weights; EMA shadow inherited
  from Stage 1 with decay switched to 0.999.
CIFAR adaptations (stated, not in the paper): unconditional -> the C "class" groups are random groups of the data;
feature stack = ConvNeXt only (ResNet-MAE dropped: 8.5 GB, ImageNet-256 specific); dropout 0 in the generator;
TF32 allowed in the FP32 ConvNeXt path.
Run: python -m src.ogen.posttrain --init <stage1_run> --out <dir>
"""
import argparse, copy, json, os, pathlib, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vu

ROOT = pathlib.Path(__file__).resolve().parents[2]
if not (ROOT / "third_party/priming/losses/matching.py").exists():
    raise SystemExit("third_party/priming is empty: run scripts/setup_third_party.sh (see THIRD_PARTY.md)")
sys.path.insert(0, str(ROOT / "third_party" / "priming"))
from models.encoders.full.vendor.convnext import ConvNextBase, _map_hf_key_to_local  # noqa: E402
from losses.matching import Matching  # noqa: E402

from .data import cifar_uint8, cifar_labels, to_float, GPUSampler, dataset_uint8, dataset_labels, CHANNELS, HFLIP  # noqa: E402
from .unet import UNet  # noqa: E402
from .train import fingerprint, ema_update  # noqa: E402

from .paths import OUT
CONVNEXT_SRC = OUT / "downloads/encoders/convnextv2-base-22k-224/pytorch_model.bin"
CALIB = OUT / "results/ot_calibration/convnext_cifar.json"


class ConvNextEncoder(torch.nn.Module):
    """ConvNeXt branch of the official PixelDriftingEncoders (models/encoders/full/encoders.py), input [0,1] BCHW."""
    family_names = ("convnext",)

    def __init__(self, device, microbatch=64):
        super().__init__()
        with torch.device("meta"):
            model = ConvNextBase(dtype=torch.float32)
        del model.head
        up = torch.load(CONVNEXT_SRC, map_location="cpu", weights_only=True)
        tgt = model.state_dict()
        state = {}
        for k, v in up.items():
            k = _map_hf_key_to_local(k)
            if k in tgt:
                assert v.shape == tgt[k].shape, k
                state[k] = v.to(tgt[k].dtype).contiguous()
        model.load_state_dict(state, strict=True, assign=True)
        self.convnext = model.to(device).eval().requires_grad_(False)
        self.microbatch = microbatch

    def encode_family(self, images, family):
        assert family == "convnext"
        with torch.autocast(images.device.type, enabled=False):
            images = images.float()
            mean = images.new_tensor([.485, .456, .406])[None, :, None, None]
            std = images.new_tensor([.229, .224, .225])[None, :, None, None]
            pixels = ((images - mean) / std).permute(0, 2, 3, 1).contiguous()
            rows = self.convnext.get_activations(pixels)
            return {f"convnext__{k}": v.flatten(1) for k, v in rows.items()}

    def forward(self, images):
        parts = [self.encode_family(p, "convnext") for p in images.split(self.microbatch)]
        return {n: torch.cat([p[n] for p in parts]) for n in parts[0]}


class ConvNextCifarEncoder(torch.nn.Module):
    """Two backbones as in the paper: ConvNeXt-V2-B (official port, run at `res` px instead of 224) + the supervised
    CIFAR-10 ResNet-18 branch of encoders.py ("cls"). The vendored code is untouched: its hard-coded 224 resize is
    redirected by wrapping F.interpolate inside that module (only when asked for (224, 224))."""

    def __init__(self, device, microbatch=64, res=112, cls_branch="cls"):
        super().__init__()
        import models.encoders.full.vendor.convnext as cvx
        from .encoders import SimCLRResNetEncoder
        _orig = F.interpolate

        def _interp(x, size=None, *args, **kw):
            return _orig(x, (res, res) if size == (224, 224) else size, *args, **kw)
        cvx.F.interpolate = _interp
        self.cx = ConvNextEncoder(device, microbatch)
        self.cls = SimCLRResNetEncoder(device, microbatch, branches=(cls_branch,))
        self.family_names = ("convnext", cls_branch)
        self.microbatch = microbatch

    def encode_family(self, images, family):
        if family == "convnext":
            return self.cx.encode_family(images.repeat(1, 3, 1, 1) if images.shape[1] == 1 else images, family)
        return self.cls.encode_family(images, family)

    def forward(self, images):
        out = {}
        for fam in self.family_names:
            parts = [self.encode_family(p, fam) for p in images.split(self.microbatch)]
            out.update({n: torch.cat([p[n] for p in parts]) for n in parts[0]})
        return out


@torch.no_grad()
def calibrate(enc, device, CALIB=CALIB, dataset="cifar10"):
    """Median pairwise Euclidean distance among 2000 train images (seed 0), per output (paper App. C)."""
    if CALIB.exists():
        return json.loads(CALIB.read_text())
    x = dataset_uint8(dataset, "train")
    idx = np.random.default_rng(0).choice(len(x), 2000, replace=False)
    y = to_float(x[idx]).to(device).mul(.5).add(.5)
    feats = enc(y)
    bw = {}
    for n, f in feats.items():
        d = torch.pdist(f.double())
        bw[n] = float(d.median())
    CALIB.parent.mkdir(parents=True, exist_ok=True)
    out = {"bandwidths": bw, "weights": {n: 1.0 for n in bw}, "n": 2000, "seed": 0}
    CALIB.write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="stage-1 run dir (uses last.pt: net=online, ema=shadow)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--refs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=125)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--coef", type=float, default=5.0)
    ap.add_argument("--microbatch", type=int, default=64)
    ap.add_argument("--gen_chunk", type=int, default=0,
                    help=">0: exact memory-saving backward -- dL/dx from the matcher on a no-grad generation, then "
                         "recompute the generator in chunks of this size and backprop dL/dx (identical gradient)")
    ap.add_argument("--dataset", choices=["cifar10", "fmnist", "mnist"], default="cifar10")
    ap.add_argument("--encoder", choices=["convnext", "simclr", "simclr+cls", "convnext112+cls", "convnext112+fcls", "convnext112+mcls"],
                    default="convnext",
                    help="simclr = CIFAR-10 SimCLR ResNet-34; simclr+cls adds a supervised CIFAR ResNet-18 branch "
                         "(src/ogen/encoders.py); both native 32px; each branch weighted 1/2 as in the paper")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_classes", type=int, default=0,
                    help="10 = class-conditional: each of the --groups OT groups is one distinct class (paper)")
    ap.add_argument("--fix_t", type=float, default=None,
                    help="init is a flow-matching net (train.py --objective fm): use it as one-step generator at this "
                         "fixed t (s = 1-t); the time input stays frozen at t")
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--ckpt_every", type=int, default=250)
    ap.add_argument("--keep_every", type=int, default=1250)
    a = ap.parse_args()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)

    if a.encoder in ("convnext112+cls", "convnext112+fcls", "convnext112+mcls"):
        br = a.encoder.split("+")[1]
        enc = ConvNextCifarEncoder(dev, a.microbatch, res=112, cls_branch=br)
        tag = "convnext112_cls_cifar" if a.dataset == "cifar10" else f"convnext112_{br}_{a.dataset}"
        cal = calibrate(enc, dev, OUT / f"results/ot_calibration/{tag}.json", dataset=a.dataset)
        per = {b: sum(n.startswith(b + "__") for n in cal["weights"]) for b in enc.family_names}
        cal["weights"] = {n: 1.0 / (2 * per[n.split("__")[0]]) for n in cal["weights"]}
    elif a.encoder.startswith("simclr"):
        from .encoders import SimCLRResNetEncoder
        br = tuple(a.encoder.split("+"))
        enc = SimCLRResNetEncoder(dev, a.microbatch, branches=br)
        cal = calibrate(enc, dev, OUT / f"results/ot_calibration/{'_'.join(br)}_cifar.json")
        # paper: weights uniform within a backbone, each backbone sums to 1/len(branches)
        per = {b: sum(n.startswith(b + "__") for n in cal["weights"]) for b in br}
        cal["weights"] = {n: 1.0 / (len(br) * per[n.split("__")[0]]) for n in cal["weights"]}
    else:
        enc = ConvNextEncoder(dev, a.microbatch)
        cal = calibrate(enc, dev)
    matcher = Matching(enc, bandwidths=cal["bandwidths"], weights=cal["weights"], kind="featurewise_ot",
                       coefficient=a.coef, global_classes=a.groups, samples=a.samples, references=a.refs,
                       epsilon_mixture={"0.001": .75, "0.01": .25}, iterations=1000, tolerance=1e-6,
                       distributed=False)

    K = a.num_classes
    assert K == 0 or a.groups <= K, "need distinct classes per update"
    CH = CHANNELS[a.dataset]
    net = UNet(dropout=0.0, num_classes=K, in_ch=CH, time_cond=a.fix_t is not None).to(dev)
    net.fixed_t = a.fix_t
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=0.0, betas=(0.9, 0.95), weight_decay=0.0)
    sampler = GPUSampler(dataset_uint8(a.dataset, "train"), dev, hflip=HFLIP[a.dataset], seed=a.seed + 1,
                         labels=dataset_labels(a.dataset, "train") if K else None)
    gcls = torch.Generator().manual_seed(a.seed + 11)
    gneg = torch.Generator(device=dev).manual_seed(a.seed + 7919)
    step = 0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev)
        net.load_state_dict(ck["net"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        sampler.g.set_state(ck["data_rng"].cpu()); gneg.set_state(ck["neg_rng"].cpu())
        torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        if "cls_rng" in ck:
            gcls.set_state(ck["cls_rng"].cpu())
        step = ck["step"]
    else:
        src = torch.load(pathlib.Path(a.init) / "last.pt", map_location=dev)
        net.load_state_dict(src["net"]); ema.load_state_dict(src["ema"])
        torch.save(dict(ema=ema.state_dict(), step=0), out / "ema_0000000.pt")
    meta = dict(vars(a), fingerprint=fingerprint(),
                n_feature_outputs=len(cal["bandwidths"]))
    print(json.dumps(meta), flush=True)
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    log = open(out / "train.log", "a")
    xfix = torch.randn((256, CH, 32, 32), generator=torch.Generator().manual_seed(0)).to(dev)
    cfix = torch.arange(256, device=dev) % K if K else None
    C, B, N = a.groups, a.samples, a.refs

    def generate(z, c=None):
        with torch.autocast("cuda", torch.bfloat16):
            img = (net(z) if c is None else net(z, c)).float() * .5 + .5
        return img.reshape(C, B, *img.shape[1:])

    t_last, acc = time.perf_counter(), []
    net.train()
    while step < a.steps:
        if K:  # C distinct classes per update; group k = class classes[k] for live, reference and real sets
            classes = torch.randperm(K, generator=gcls)[:C].to(dev)
            y = (sampler.per_class(classes.tolist(), N) * .5 + .5).reshape(C, N, CH, 32, 32)
            lab = classes.repeat_interleave(B)
        else:
            y = (sampler(C * N) * .5 + .5).reshape(C, N, CH, 32, 32)
            lab = None
        with torch.no_grad():
            x_prime = generate(torch.randn((C * B, CH, 32, 32), device=dev, generator=gneg), lab)
        z = torch.randn((C * B, CH, 32, 32), device=dev)
        if a.gen_chunk:
            with torch.no_grad():
                x = generate(z, lab)
            x.requires_grad_(True)
            loss, m = matcher(x=x, x_prime=x_prime, y=y)
            loss.backward()                                  # -> x.grad = dL/dx (matcher's own input VJP)
            gx = x.grad.flatten(0, 1)
            for i in range(0, C * B, a.gen_chunk):
                sl = slice(i, i + a.gen_chunk)
                with torch.autocast("cuda", torch.bfloat16):
                    img = (net(z[sl]) if lab is None else net(z[sl], lab[sl])).float() * .5 + .5
                img.backward(gx[sl])
        else:
            x = generate(z, lab)
            loss, m = matcher(x=x, x_prime=x_prime, y=y)
            loss.backward()
        lr = a.lr * step / a.warmup if a.warmup > 0 and step <= a.warmup else a.lr  # official zero-based schedule
        for g in opt.param_groups:
            g["lr"] = lr
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite grad norm")
        opt.step()
        ema_update(ema, net, a.ema)
        opt.zero_grad(set_to_none=True)
        step += 1
        acc.append(dict(loss=loss.item(), loss_cross=m["loss_cross"].item(), loss_self=m["loss_self"].item(),
                        eff_targets=m["effective_targets_mean"].item(), sk_iters=m["sinkhorn_iterations"].item(),
                        grad_norm=norm.item()))
        if step % a.log_every == 0:
            now = time.perf_counter()
            rec = {k: float(np.mean([r[k] for r in acc])) for k in acc[0]}
            rec.update(step=step, lr=lr, step_ms=1e3 * (now - t_last) / len(acc))
            t_last, acc = now, []
            if step % (10 * a.log_every) == 0:
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    g = (ema(xfix) if K == 0 else ema(xfix, cfix)).float()
                rec["ema_gen_pix_std"] = g.std(0).mean().item()
                n = 100 if K else 64
                vu.save_image(g[:n].add(1).div(2).clamp(0, 1), out / f"grid_{step:07d}.png", nrow=10 if K else 8)
            log.write(json.dumps(rec) + "\n"); log.flush()
        if step % a.ckpt_every == 0 or step == a.steps:
            ck = dict(net=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step,
                      data_rng=sampler.g.get_state(), neg_rng=gneg.get_state(), cuda_rng=torch.cuda.get_rng_state(),
                      cls_rng=gcls.get_state())
            torch.save(ck, out / "last.pt.tmp"); os.replace(out / "last.pt.tmp", out / "last.pt")
        if step % a.keep_every == 0 or step == a.steps:
            torch.save(dict(ema=ema.state_dict(), step=step), out / f"ema_{step:07d}.pt")
    (out / "DONE").write_text(str(step))


if __name__ == "__main__":
    main()
