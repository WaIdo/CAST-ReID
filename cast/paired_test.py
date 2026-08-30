#!/usr/bin/env python3
"""Paired bootstrap significance test between two Simple-CCReID checkpoints.

Both models rank the SAME queries, so per-query paired comparison cancels the
shared eval noise that drowns small deltas on low-base protocols (MEVID CC sits
at ~4-5% R1 for R50-class methods). Reports per-protocol mean delta, bootstrap
95% CI, and a two-sided p-value over query resamples.

Single-GPU, no torch.distributed. Run when the GPU is free (~6 min per model).

Usage:
  python paired_test.py --cfg configs/mevid_cal.yaml --dataset mevid \\
      --ckpt-a logs_mevid_cal_gt/.../best_model.pth.tar \\
      --ckpt-b logs_mevid_ce_only/.../best_model.pth.tar
"""
import argparse
import os
import sys

import collections
import collections.abc

# Simple-CCReID targets py<3.10; its transforms still use collections.Iterable.
# main.py installs this shim, but this script bypasses main.py entirely.
for _n in ("Iterable", "Mapping", "Sequence"):
    if not hasattr(collections, _n):
        setattr(collections, _n, getattr(collections.abc, _n))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.environ.get('CCREID_ROOT', './Simple-CCReID'))


def build_everything(cfg_path, dataset_name):
    class A:
        cfg = cfg_path
        root = os.environ.get('DATA_ROOT', './datasets')
        dataset = dataset_name
        output = '/tmp/paired_test_out'
        resume = amp = eval = tag = None
        gpu = '0'
    from configs.default_vid import get_vid_config
    config = get_vid_config(A())
    import data as D
    dataset = D.build_dataset(config)
    _, st_test, _, _ = D.build_vid_transforms(config)
    from data.dataset_loader import VideoDataset
    q_loader = DataLoader(VideoDataset(dataset.recombined_query, st_test, None),
                          batch_size=config.DATA.TEST_BATCH, shuffle=False,
                          num_workers=config.DATA.NUM_WORKERS, pin_memory=True)
    g_loader = DataLoader(VideoDataset(dataset.recombined_gallery, st_test, None),
                          batch_size=config.DATA.TEST_BATCH, shuffle=False,
                          num_workers=config.DATA.NUM_WORKERS, pin_memory=True)
    return config, dataset, q_loader, g_loader


@torch.no_grad()
def tracklet_features(model, loader, vid2clip):
    feats, pids, camids, clothes = [], [], [], []
    clip_f, clip_meta = [], []
    for vids, p, cam, cl in loader:
        f = model(vids.cuda())
        clip_f.append(f.cpu())
        clip_meta.append(torch.stack([p, cam, cl], 1))
    clip_f = torch.cat(clip_f)
    clip_meta = torch.cat(clip_meta)
    out = torch.zeros(len(vid2clip), clip_f.size(1))
    meta = torch.zeros(len(vid2clip), 3)
    for i, (s, e) in enumerate(vid2clip):
        out[i] = F.normalize(clip_f[s:e].mean(0), p=2, dim=0)
        meta[i] = clip_meta[s]
    return out, meta[:, 0].numpy(), meta[:, 1].numpy(), meta[:, 2].numpy()


def per_query_metrics(distmat, q_pids, g_pids, q_cams, g_cams, q_cl, g_cl, mode):
    """Return per-query AP and rank-1 arrays; NaN when a query has no target."""
    order = np.argsort(distmat, axis=1)
    n_q = distmat.shape[0]
    APs, R1s = np.full(n_q, np.nan), np.full(n_q, np.nan)
    for i in range(n_q):
        same_pid = g_pids == q_pids[i]
        same_cam = g_cams == q_cams[i]
        same_cl = g_cl == q_cl[i]
        if mode == 'CC':
            good = same_pid & ~same_cam & ~same_cl
            junk = same_pid & (same_cam | same_cl)
        elif mode == 'SC':
            good = same_pid & ~same_cam & same_cl
            junk = same_pid & (same_cam | ~same_cl)
        else:
            good = same_pid & ~same_cam
            junk = same_pid & same_cam
        if not good.any():
            continue
        idx = order[i][~junk[order[i]]]
        hits = good[idx]
        rows = np.where(hits)[0]
        R1s[i] = 1.0 if hits[0] else 0.0
        precision = (np.arange(len(rows)) + 1) / (rows + 1)
        APs[i] = precision.mean()
    return APs, R1s


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """Paired bootstrap over queries for mean(a) - mean(b); NaNs dropped pairwise."""
    keep = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[keep], b[keep]
    d = a - b
    rng = np.random.RandomState(seed)
    n = len(d)
    boots = np.array([d[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return d.mean(), lo, hi, min(p, 1.0), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--dataset', default='mevid')
    ap.add_argument('--ckpt-a', required=True, help='e.g. CAL-GT best_model')
    ap.add_argument('--ckpt-b', required=True, help='e.g. CE-only best_model')
    ap.add_argument('--n-boot', type=int, default=10000)
    args = ap.parse_args()

    config, dataset, q_loader, g_loader = build_everything(args.cfg, args.dataset)
    from models import build_model

    results = {}
    for tag, ckpt in [('A', args.ckpt_a), ('B', args.ckpt_b)]:
        model, _, _ = build_model(config, dataset.num_train_pids, dataset.num_train_clothes)
        state = torch.load(ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        model = model.cuda().eval()
        qf, qp, qc, qcl = tracklet_features(model, q_loader, dataset.query_vid2clip_index)
        gf, gp, gc, gcl = tracklet_features(model, g_loader, dataset.gallery_vid2clip_index)
        dist = (-torch.mm(qf, gf.t())).numpy()
        results[tag] = {m: per_query_metrics(dist, qp, gp, qc, gc, qcl, gcl, m)
                        for m in ['general', 'SC', 'CC']}
        print('[%s] %s  epoch=%s rank1(saved)=%.4f' % (tag, ckpt, state.get('epoch'), state.get('rank1', float('nan'))))
        del model
        torch.cuda.empty_cache()

    print()
    print('Paired bootstrap A-B (%d resamples):' % args.n_boot)
    for m in ['general', 'SC', 'CC']:
        for name, k in [('mAP', 0), ('R1 ', 1)]:
            d, lo, hi, p, n = paired_bootstrap(results['A'][m][k], results['B'][m][k], args.n_boot)
            sig = 'SIGNIFICANT' if p < 0.05 else 'not significant'
            print('  %-7s %s  delta=%+.4f  95%%CI=[%+.4f, %+.4f]  p=%.4f  n=%d  -> %s'
                  % (m, name, d, lo, hi, p, n, sig))


if __name__ == '__main__':
    main()
