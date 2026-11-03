"""Epoch-level noise/data pairing, decoupled from the training batch size.

Each epoch e: every train image gets a random hflip, and the n_c (=5000) images of each class c are randomly split
into chunks of <= `chunk` (default 2500) images; each chunk is paired with fresh N(0,I) noises -- exactly OT-matched
(Hungarian on ||x-y||^2 over the chunk, kind='ot') or as drawn (kind='ind'). Chunking: exact Hungarian on real
CIFAR classes costs 0.4 s / 4 s / 36 s for 1250 / 2500 / 5000 points while the cost reduction is 4.04 / 4.16 / 4.38 %. The 50k pairs are shuffled and consumed in training batches. Both arms share this
pipeline (same noise, flips and order RNG); only the permutation differs.
Noise and the per-class cost matrices are made on the GPU; the Hungarian solves for epoch e+1 run in spawned CPU
worker processes while epoch e trains.
"""
import concurrent.futures as cf
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _lsa(cost):
    r, col = linear_sum_assignment(cost.astype(np.float64))
    return col.astype(np.int64), float(np.trace(cost)) / len(cost), float(cost[r, col].mean())


class EpochPairs:
    def __init__(self, x_gpu, labels, kind, seed=0, workers=20, chunk=2500, hflip=True):
        """x_gpu: (N,3,32,32) float32 on GPU in [-1,1]; labels: (N,) long (any device)."""
        self.x, self.kind, self.seed, self.dev = x_gpu, kind, seed, x_gpu.device
        labels = labels.to(self.dev)
        self.K = int(labels.max()) + 1
        self.cls_idx = [torch.nonzero(labels == c)[:, 0] for c in range(self.K)]
        self.N = len(labels)
        self.chunk = chunk
        self.hflip = hflip
        self.pool = (cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))
                     if kind == "ot" else None)
        self._pending = {}
        self.stats = {}

    def _draw(self, epoch):
        g = torch.Generator(device=self.dev).manual_seed(self.seed * 1_000_003 + epoch)
        noise = torch.randn((self.N, self.x[0].numel()), device=self.dev, generator=g)  # row k pairs with class slot k
        flips = torch.rand(self.N, device=self.dev, generator=g) < 0.5            # per image index
        if not self.hflip:                       # same RNG draw (comparable streams), flips disabled
            flips = torch.zeros_like(flips)
        order = torch.randperm(self.N, device=self.dev, generator=g)
        return noise, flips, order

    def prefetch(self, epoch):
        if epoch in self._pending:
            return
        noise, flips, order = self._draw(epoch)
        g = torch.Generator(device=self.dev).manual_seed(self.seed * 7919 + epoch)
        chunks, off = [], 0          # (class, image indices of chunk, noise row offset, future|None)
        for c in range(self.K):
            idx = self.cls_idx[c][torch.randperm(len(self.cls_idx[c]), device=self.dev, generator=g)]
            for part in idx.split(self.chunk):
                fut = None
                if self.kind == "ot":
                    y = self.x[part]
                    y = torch.where(flips[part][:, None, None, None], y.flip(-1), y).flatten(1)
                    x = noise[off:off + len(part)]
                    cost = (x.pow(2).sum(1)[:, None] + y.pow(2).sum(1)[None] - 2 * x @ y.T).float().cpu().numpy()
                    fut = self.pool.submit(_lsa, cost)
                chunks.append((c, part, off, fut))
                off += len(part)
        self._pending[epoch] = (noise, flips, order, chunks)

    def epoch(self, epoch):
        """Returns (noise (N,3072), image index (N,), flip (N,), label (N,)) on GPU, in shuffled order."""
        self.prefetch(epoch)
        noise, flips, order, chunks = self._pending.pop(epoch)
        img, lab, red = [], [], []
        for c, part, off, fut in chunks:     # chunks are in noise-row order
            if fut is not None:
                col, ic, oc = fut.result()
                img.append(part[torch.from_numpy(col).to(self.dev)])
                red.append(1 - oc / ic)
            else:
                img.append(part)
            lab.append(torch.full((len(part),), c, device=self.dev))
        img, lab = torch.cat(img), torch.cat(lab)
        if red:
            self.stats = {"ot_cost_reduction": float(np.mean(red))}
        return noise[order], img[order], flips[img[order]], lab[order]
