"""Fixed-noise (s=0.8) x0-denoiser on CIFAR-10 with independent vs minibatch-OT coupling.

Loss: || D(0.8 X + 0.2 Y_pi) - Y_pi ||^2 (plain pixel MSE, mean over pixels).
Run:  python -m src.ogen.train --coupling ot --bs 128 --steps 200000 --out <run_dir>

--objective fm (2026-10-28): class-conditional flow matching with the same U-Net + DDPM time embedding, x-prediction
trained with the velocity loss. Z_t = (1-t) X + t Y (t=1 data; s = 1-t), t ~ U[0,1] per sample, v = Y - X,
v_hat = (D(Z_t,t) - Z_t)/(1-t)  =>  ||v_hat - v||^2 = ||D - Y||^2 / (1-t)^2, weight clipped at (1-t) >= --fm_tclip
(JiT-style). Coupling/pairing unchanged (ind or per-epoch pool OT). val/grids use D at fixed t = 1 - --noise.
--fm_loss percep (2026-10-29): the per-sample pixel MSE ||D-Y||^2 is replaced by the Stage-2 feature space:
sum_k w_k ||phi_k(D) - phi_k(Y)||^2 / sigma_k^2 over the 26 ConvNeXt112+cls outputs (same calibrated sigma_k and
backbone weights 1/2 as posttrain.py), same t-weight 1/max(1-t, tclip)^2. Not a proper velocity loss any more.
--fm_pix_w w (with percep): adds the pixel term as one more "output": w ||D-Y||^2 / sigma_pix^2 (images in [0,1],
sigma_pix = median pairwise distance of the 2000 calibration train images, seed 0), i.e. weight w vs 1/2 per backbone.
"""
import argparse, copy, hashlib, json, math, os, pathlib, time
import torch
import torchvision.utils as vu

from .data import cifar_uint8, cifar_labels, to_float, GPUSampler, dataset_uint8, dataset_labels, CHANNELS, HFLIP
from .unet import UNet
from .coupling import couple, interp, ot_perm, ot_perm_within
from .pairing import EpochPairs

SRC = pathlib.Path(__file__).parent


def fingerprint():
    h = hashlib.md5()
    for f in sorted(SRC.glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:10]


@torch.no_grad()
def ema_update(ema, net, decay):
    for pe, p in zip(ema.parameters(), net.parameters()):
        pe.lerp_(p, 1 - decay)
    for be, b in zip(ema.buffers(), net.buffers()):
        be.copy_(b)


@torch.no_grad()
def batched(net, x, bs=500, c=None):
    if c is None:
        return torch.cat([net(x[i:i + bs]) for i in range(0, len(x), bs)])
    return torch.cat([net(x[i:i + bs], c[i:i + bs]) for i in range(0, len(x), bs)])


@torch.no_grad()
def val_metrics(net, val, amp, s=0.8):
    """Held-out denoising MSE (independent and OT-paired test pairs) + one-step output stats."""
    out = {}
    with torch.autocast("cuda", torch.bfloat16, enabled=amp):
        for k in ("ind", "ot"):
            z = interp(val["x"], val[f"y_{k}"], s)
            d = batched(net, z, c=val.get("c")).float()
            out[f"val_mse_{k}"] = (d - val[f"y_{k}"]).pow(2).mean().item()
        g = batched(net, val["x_gen"], c=val.get("c_gen")).float()
    out["gen_pix_std"] = g.std(0).mean().item()          # across-sample per-pixel std
    out["gen_mean_abs"] = g.abs().mean().item()
    out["gen_pdist"] = torch.pdist(g[:256].flatten(1)).mean().item()
    return out, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coupling", choices=["ind", "ot"], required=True)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--keep_every", type=int, default=50000)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--num_classes", type=int, default=0, help="10 = class-conditional (OT within class)")
    ap.add_argument("--noise", type=float, default=0.8, help="s in Z = s X + (1-s) Y; 1.0 = raw noise input")
    ap.add_argument("--dataset", choices=["cifar10", "fmnist", "mnist"], default="cifar10")
    ap.add_argument("--ot_chunk", type=int, default=2500, help="epoch pairing: OT pool size per class chunk")
    ap.add_argument("--pairing", choices=["batch", "epoch"], default="batch",
                    help="epoch: per epoch, exact OT per class over the whole class (decoupled from --bs); needs --num_classes")
    ap.add_argument("--objective", choices=["fixed", "fm"], default="fixed",
                    help="fixed: x0-MSE at s=--noise (Stage 1); fm: flow matching over t~U[0,1] (time-conditioned)")
    ap.add_argument("--fm_tclip", type=float, default=0.05, help="fm: velocity-loss weight 1/max(1-t, tclip)^2")
    ap.add_argument("--fm_loss", choices=["mse", "percep"], default="mse",
                    help="fm: per-sample distance D vs Y: pixel MSE or Stage-2 feature distance (convnext112+cls)")
    ap.add_argument("--enc_microbatch", type=int, default=64)
    ap.add_argument("--fm_pix_w", type=float, default=0.0, help="percep: weight of the added pixel term (0 = none)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fm = a.objective == "fm"
    amp = not a.no_amp
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cond = a.num_classes > 0
    CH = CHANNELS[a.dataset]
    sampler = GPUSampler(dataset_uint8(a.dataset, "train"), dev, hflip=HFLIP[a.dataset], seed=a.seed,
                         labels=dataset_labels(a.dataset, "train") if cond else None)
    # fixed held-out set: 2000 test images, fixed noise; OT pairs = Hungarian in fixed chunks of 128 (same for every run)
    g = torch.Generator().manual_seed(1234)
    yt = to_float(dataset_uint8(a.dataset, "test")[:2000]).to(dev)
    xv = torch.randn(yt.shape, generator=g).to(dev)
    if cond:
        lt = dataset_labels(a.dataset, "test")[:2000].to(dev)
        perm = torch.cat([ot_perm_within(xv[i:i + 128], yt[i:i + 128], lt[i:i + 128])[0] + i
                          for i in range(0, len(yt), 128)])
    else:
        perm = torch.cat([ot_perm(xv[i:i + 128], yt[i:i + 128])[0] + i for i in range(0, len(yt), 128)])
    val = {"x": xv, "y_ind": yt, "y_ot": yt[perm],
           "x_gen": torch.randn((1024, CH, 32, 32), generator=torch.Generator().manual_seed(0)).to(dev)}
    if cond:  # labels unchanged by within-class OT; generation grid class-balanced (row = class for nrow=10 grids)
        val["c"], val["c_gen"] = lt, torch.arange(1024, device=dev) % a.num_classes

    if a.fm_loss == "percep":
        assert fm and a.dataset == "cifar10"
        from .posttrain import ConvNextCifarEncoder, calibrate, OUT as _OUT
        enc = ConvNextCifarEncoder(dev, a.enc_microbatch, res=112, cls_branch="cls")
        cal = calibrate(enc, dev, _OUT / "results/ot_calibration/convnext112_cls_cifar.json")
        per = {b: sum(n.startswith(b + "__") for n in cal["bandwidths"]) for b in enc.family_names}
        fw = {n: 1.0 / (2 * per[n.split("__")[0]] * cal["bandwidths"][n] ** 2) for n in cal["bandwidths"]}

        import numpy as _np
        _xc = dataset_uint8(a.dataset, "train")
        _xc = to_float(_xc[_np.random.default_rng(0).choice(len(_xc), 2000, replace=False)]).to(dev).mul(.5).add(.5)
        sig_pix = float(torch.pdist(_xc.flatten(1).double()).median()); del _xc

        def percep(d, y):  # per-sample weighted squared feature distance; images to [0,1] as in Stage 2
            d01, y01 = d.float() * .5 + .5, y * .5 + .5
            fd = enc(d01)
            with torch.no_grad():
                fy = enc(y01)
            out = sum(fw[n] * (fd[n] - fy[n]).pow(2).sum(1) for n in fd)
            if a.fm_pix_w:
                out = out + a.fm_pix_w * (d01 - y01).pow(2).flatten(1).sum(1) / sig_pix ** 2
            return out
    net = UNet(dropout=a.dropout, num_classes=a.num_classes, in_ch=CH, time_cond=fm).to(dev)
    if fm:
        net.fixed_t = 1 - a.noise   # val metrics / grids: one-step D at s = --noise
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.999), weight_decay=a.wd)
    sched = lambda s: min(1.0, (s + 1) / a.warmup)
    step = 0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev)
        net.load_state_dict(ck["net"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        sampler.g.set_state(ck["data_rng"].cpu())
        torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        step = ck["step"]
    fwd = torch.compile(net) if a.compile else net
    meta = dict(vars(a), fingerprint=fingerprint(), params=sum(p.numel() for p in net.parameters()),)
    print(json.dumps(meta), flush=True)
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    log = open(out / "train.log", "a")

    acc = dict(loss=0.0, ot_time=0.0, step_time=0.0, n=0)
    t_last = time.perf_counter()
    if a.pairing == "epoch":
        assert cond, "epoch pairing is per class"
        pairer = EpochPairs(sampler.x, dataset_labels(a.dataset, "train"), a.coupling, seed=a.seed, chunk=a.ot_chunk,
                            hflip=HFLIP[a.dataset])
        spe = sampler.x.shape[0] // a.bs
        cur_ep, cur = -1, None
    while step < a.steps:
        for pg in opt.param_groups:
            pg["lr"] = a.lr * sched(step)
        if a.pairing == "epoch":
            ep, pos = divmod(step, spe)
            t0 = time.perf_counter()
            if ep != cur_ep:
                cur = pairer.epoch(ep)
                pairer.prefetch(ep + 1)
                cur_ep = ep
                log.write(json.dumps(dict(step=step, kind="epoch", epoch=ep, **pairer.stats)) + "\n"); log.flush()
            sl = slice(pos * a.bs, (pos + 1) * a.bs)
            x = cur[0][sl].view(-1, CH, 32, 32)
            y = sampler.x[cur[1][sl]]
            y = torch.where(cur[2][sl][:, None, None, None], y.flip(-1), y)
            lab = cur[3][sl]
            ot_dt = time.perf_counter() - t0
        else:
            y = sampler(a.bs)
            lab = sampler.last_labels if cond else None
            x = torch.randn_like(y)
            y, ot_dt = couple(x, y, a.coupling, lab)
        if fm:
            t = torch.rand(len(y), device=dev)
            z = interp(x, y, (1 - t)[:, None, None, None])
            with torch.autocast("cuda", torch.bfloat16, enabled=amp):
                d = fwd(z, lab, t)
            w = (1 - t).clamp_min(a.fm_tclip).pow(-2)
            dist = percep(d, y) if a.fm_loss == "percep" else (d.float() - y).pow(2).mean((1, 2, 3))
            loss = (w * dist).mean()
        else:
            z = interp(x, y, a.noise)
            with torch.autocast("cuda", torch.bfloat16, enabled=amp):
                d = fwd(z) if lab is None else fwd(z, lab)
            loss = (d.float() - y).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
        opt.step()
        # EMA warmup (as in many DDPM ports): decay ramps up early
        ema_update(ema, net, min(a.ema, (1 + step) / (10 + step)))
        step += 1
        acc["loss"] += loss.item(); acc["ot_time"] += ot_dt; acc["n"] += 1
        if step % a.log_every == 0:
            now = time.perf_counter()
            rec = dict(step=step, loss=acc["loss"] / acc["n"], ot_ms=1e3 * acc["ot_time"] / acc["n"],
                       step_ms=1e3 * (now - t_last) / acc["n"], lr=opt.param_groups[0]["lr"])
            t_last = now
            acc = dict(loss=0.0, ot_time=0.0, n=0)
            log.write(json.dumps(rec) + "\n"); log.flush()
        if step % a.eval_every == 0 or step == a.steps:
            net.eval()
            m_ema, gen = val_metrics(ema, val, amp, a.noise)
            m_net, _ = val_metrics(net, val, amp, a.noise)
            net.train()
            rec = dict(step=step, kind="val", **{f"ema_{k}": v for k, v in m_ema.items()},
                       **{f"net_{k}": v for k, v in m_net.items()})
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
            n = 100 if cond else 64
            vu.save_image(gen[:n].add(1).div(2).clamp(0, 1), out / f"grid_{step:07d}.png", nrow=10 if cond else 8)
        if step % a.ckpt_every == 0 or step == a.steps:
            ck = dict(net=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step,
                      data_rng=sampler.g.get_state(), cuda_rng=torch.cuda.get_rng_state(), args=vars(a))
            torch.save(ck, out / "last.pt.tmp"); os.replace(out / "last.pt.tmp", out / "last.pt")
            if step % a.keep_every == 0 or step == a.steps:
                torch.save(dict(ema=ema.state_dict(), step=step), out / f"ema_{step:07d}.pt")
    (out / "DONE").write_text(str(step))


if __name__ == "__main__":
    main()
