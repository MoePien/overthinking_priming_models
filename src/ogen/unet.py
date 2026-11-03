"""DDPM-style CIFAR-10 U-Net (Ho et al. 2020 / ncsnpp-lite conventions) WITHOUT time conditioning.

ch=128, ch_mult=(1,2,2,2), 2 res blocks per level, self-attention at 16x16, GroupNorm(32), SiLU,
dropout 0.1, zero-init output convs. ~35.7M params (same as DDPM CIFAR minus the time MLP).
Fixed noise level s=0.8 -> no time input, as in Priming Models' simplest instantiation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(c):
    return nn.GroupNorm(32, c)


def zero(m):
    nn.init.zeros_(m.weight)
    nn.init.zeros_(m.bias)
    return m


class ResBlock(nn.Module):
    def __init__(self, cin, cout, dropout, emb_dim=0):
        super().__init__()
        self.n1, self.c1 = gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        if emb_dim:  # class conditioning, injected like the DDPM time embedding
            self.emb = nn.Linear(emb_dim, cout)
        self.n2, self.c2 = gn(cout), zero(nn.Conv2d(cout, cout, 3, padding=1))
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb=None):
        h = self.c1(F.silu(self.n1(x)))
        if emb is not None:
            h = h + self.emb(emb)[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return self.skip(x) + h


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n = gn(c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.out = zero(nn.Conv2d(c, c, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, C, H * W).transpose(-1, -2).unbind(1)
        h = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0]
        return x + self.out(h.transpose(-1, -2).reshape(B, C, H, W))


def timestep_embedding(t, dim):
    """DDPM sinusoidal embedding of t in [0,1] (scaled by 1000 like the DDPM integer timesteps)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = 1000 * t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], 1)


class UNet(nn.Module):
    def __init__(self, ch=128, ch_mult=(1, 2, 2, 2), num_res=2, attn_res=(16,), dropout=0.1, res=32, num_classes=0,
                 in_ch=3, time_cond=False):
        super().__init__()
        self.num_classes, self.time_cond, self.ch = num_classes, time_cond, ch
        self.fixed_t = None  # time-conditioned nets: t used when forward() gets none (one-step use at a fixed t)
        E = 4 * ch if (num_classes or time_cond) else 0
        if num_classes:
            self.cemb = nn.Sequential(nn.Embedding(num_classes, E), nn.SiLU(), nn.Linear(E, E))
        if time_cond:  # flow-matching net (src/ogen/train.py --objective fm): DDPM time MLP, added to the class emb
            self.temb = nn.Sequential(nn.Linear(ch, E), nn.SiLU(), nn.Linear(E, E))
        self.inp = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.down = nn.ModuleList()
        chans, c, r = [ch], ch, res
        for i, m in enumerate(ch_mult):
            for _ in range(num_res):
                blk = nn.ModuleList([ResBlock(c, ch * m, dropout, E)])
                c = ch * m
                if r in attn_res:
                    blk.append(Attn(c))
                self.down.append(blk)
                chans.append(c)
            if i != len(ch_mult) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(c, c, 3, stride=2, padding=1)]))
                chans.append(c)
                r //= 2
        self.mid = nn.ModuleList([ResBlock(c, c, dropout, E), Attn(c), ResBlock(c, c, dropout, E)])
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(ch_mult))):
            for _ in range(num_res + 1):
                blk = nn.ModuleList([ResBlock(c + chans.pop(), ch * m, dropout, E)])
                c = ch * m
                if r in attn_res:
                    blk.append(Attn(c))
                self.up.append(blk)
            if i != 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"),
                                              nn.Conv2d(c, c, 3, padding=1)]))
                r *= 2
        self.out = nn.Sequential(gn(c), nn.SiLU(), zero(nn.Conv2d(c, in_ch, 3, padding=1)))

    def forward(self, x, c=None, t=None):
        e = self.cemb(c) if self.num_classes else None
        if self.time_cond:
            if t is None:
                t = torch.full((len(x),), self.fixed_t, device=x.device)
            te = self.temb(timestep_embedding(t, self.ch))
            e = te if e is None else e + te
        run = lambda m, h: m(h, e) if isinstance(m, ResBlock) else m(h)
        h = self.inp(x)
        hs = [h]
        for blk in self.down:
            if isinstance(blk[0], ResBlock):
                for m in blk:
                    h = run(m, h)
            else:
                h = blk[0](h)
            hs.append(h)
        for m in self.mid:
            h = run(m, h)
        for blk in self.up:
            if isinstance(blk[0], ResBlock):
                h = run(blk[0], torch.cat([h, hs.pop()], 1))
                for m in blk[1:]:
                    h = run(m, h)
            else:
                h = blk[1](blk[0](h))
        return self.out(h)


if __name__ == "__main__":
    net = UNet()
    print(sum(p.numel() for p in net.parameters()) / 1e6, "M params")
    print(net(torch.randn(2, 3, 32, 32)).shape)
    netc = UNet(num_classes=10)
    print(sum(p.numel() for p in netc.parameters()) / 1e6, "M params (cond)", netc(torch.randn(2, 3, 32, 32), torch.tensor([1, 2])).shape)
    nett = UNet(num_classes=10, time_cond=True)
    print(sum(p.numel() for p in nett.parameters()) / 1e6, "M params (cond+time)",
          nett(torch.randn(2, 3, 32, 32), torch.tensor([1, 2]), torch.tensor([0.1, 0.9])).shape)
