#!/usr/bin/env python3
"""Uniform model soup: average the weights of N same-recipe checkpoints.

Fixes vs v1 (2026-08-08 21:50):
  - torch>=2.6 defaults weights_only=True and refuses the numpy scalars inside
    Simple-CCReID checkpoints -> load with weights_only=False (our own files).
  - main.py resumes with a BARE torch.load (weights_only=True on torch 2.9),
    so the soup file must contain only tensors + python natives: we rebuild a
    sanitized dict with exactly the keys main.py needs (model/classifier/
    clothes_classifier state dicts, int epoch, float rank1).

Usage: soup_ckpt.py out.pth in1.pth in2.pth in3.pth
"""
import sys

import torch

out, paths = sys.argv[1], sys.argv[2:]
assert len(paths) >= 2, 'need at least two checkpoints'
ckpts = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]

result = {}
for key in ('model_state_dict', 'classifier_state_dict', 'clothes_classifier_state_dict'):
    if key not in ckpts[0]:
        print('missing key in ckpt:', key)
        continue
    sd = {}
    n_avg = 0
    for k, v in ckpts[0][key].items():
        if torch.is_tensor(v) and v.is_floating_point():
            sd[k] = torch.stack([c[key][k].float() for c in ckpts]).mean(0).to(v.dtype)
            n_avg += 1
        else:
            sd[k] = v
    result[key] = sd
    print('%s: averaged %d tensors over %d ckpts' % (key, n_avg, len(ckpts)))

result['epoch'] = int(ckpts[0].get('epoch', 0))
result['rank1'] = float(ckpts[0].get('rank1', 0.0))
torch.save(result, out)
print('soup saved:', out)
