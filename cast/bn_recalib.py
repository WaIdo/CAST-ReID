#!/usr/bin/env python3
"""Deterministic BatchNorm recalibration for a model-soup checkpoint.

Uniform weight averaging invalidates BN running statistics (the averaged
buffers do not describe the averaged weights' activations). The CLIP visual
tower is LayerNorm-only, but the video wrapper adds a BatchNorm neck before
the classifier, so ViT-B/16 does carry running statistics; the deployed
ViT-B/16 soup nonetheless ships without recalibration. ResNet-50 soups drop
10+ mAP without this step.

Protocol (fixed, part of the method definition): reset every BN buffer, run
one pass over the training set with the test-time spatial transform and a
TemporalDivisionCrop(seq_len) temporal transform, cumulative-average momentum,
fixed order (shuffle seeded), then write the buffers back into the checkpoint.

Usage: bn_recalib29.py --cfg CFG --dataset ccvid|mevid --soup IN.pth --out OUT.pth
"""
import collections, collections.abc
for _n in ("Iterable", "Mapping", "Sequence", "Callable"):
    if not hasattr(collections, _n):
        setattr(collections, _n, getattr(collections.abc, _n))

import argparse, os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CC = os.environ.get("CCREID_ROOT", "./Simple-CCReID")
sys.path.insert(0, CC)
os.chdir(CC)

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--soup", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

from configs.default_vid import get_vid_config
import data as D
from data import temporal_transforms as TT
from data.dataset_loader import VideoDataset
from models import build_model


class Args:
    cfg = a.cfg
    root = os.environ.get("DATA_ROOT", "./datasets")
    dataset = a.dataset
    output = "/tmp/bn_recalib29"
    resume = amp = eval = tag = None
    gpu = "0"


config = get_vid_config(Args())
dataset = D.build_dataset(config)
_, st_test, _, _ = D.build_vid_transforms(config)
tt = TT.TemporalDivisionCrop(size=config.AUG.SEQ_LEN)
train_data = VideoDataset(dataset.train, st_test, tt)
g = torch.Generator(); g.manual_seed(20260820)
loader = DataLoader(train_data, batch_size=16, shuffle=True, generator=g,
                    num_workers=4, pin_memory=True, drop_last=False)
print(f"calib set: {len(train_data)} tracklets, {len(loader)} batches")

model, _, _ = build_model(config, dataset.num_train_pids, dataset.num_train_clothes)
ckpt = torch.load(a.soup, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model = model.cuda()

n_bn = 0
for m in model.modules():
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.running_mean.zero_()
        m.running_var.fill_(1.0)
        m.num_batches_tracked.zero_()
        m.momentum = None
        n_bn += 1
print(f"reset {n_bn} BN layers")

model.train()
with torch.no_grad():
    for batch in loader:
        _ = model(batch[0].cuda())

sd = model.state_dict()
for k in ckpt["model_state_dict"]:
    if ("running_mean" in k or "running_var" in k or "num_batches_tracked" in k) and k in sd:
        ckpt["model_state_dict"][k] = sd[k].cpu()

torch.save(ckpt, a.out)
print(f"BN-calibrated soup saved: {a.out}")
