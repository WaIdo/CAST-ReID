#!/usr/bin/env python3
"""CAST: turn caption-derived pseudo outfits into soft, confidence-calibrated supervision.

Hard cluster ids throw away everything the clustering knew: a tracklet that matched
its cluster at Jaccard 0.95 and one that barely cleared the threshold at 0.71 become
the same one-hot. At purity 0.842 that silently feeds ~16% wrong one-hots into CAL's
adversarial target. This script keeps the graded information.

Outputs (npz), all indexed by pseudo-outfit id so the training side needs no
dataloader change -- the cluster is the unit of supervision:
  S [C,C]  cluster-to-cluster caption similarity (symmetric, unit diagonal)
  Q [C,C]  soft target distribution per cluster: where its members' evidence
           actually points, i.e. which other pseudo-outfits it is confusable with
  w [C]    confidence, two modes (--w-mode):
           margin (legacy): top-1 vs top-2 margin of Q. Flawed under
             over-segmentation: a pure cluster split from its wording-twin gets
             margin~0 and is silenced although its adversarial anchor is fine.
             2026-08-04 diagnosis: 807/960 MEVID clusters silenced, halving the
             net adversarial contribution (+3.5/+2.8 vs hard pseudo +10.1/+5.5).
           coherence (the fix): members' mean pairwise caption Jaccard --
             measures whether the cluster's own evidence agrees with itself,
             indifferent to twins; singletons get the global mean.
  drift [C,C]  1 - S, the caption-drift weight used by the cross-outfit alignment

With --agree-from (a multi-window captions jsonl from caption_multiwindow.py), w is
additionally scaled by each cluster's mean clothes-field vote agreement -- the C3
channel: a cluster whose members' captions were unstable across temporal windows is
unreliable regardless of how cleanly it separated from other clusters.

Usage:
  python soft_pseudo.py --captions cast/ccvid_captions.jsonl \\
      --pseudo cast/pseudo_ccvid_th07_v2.json --out cast/soft_ccvid_th07.npz
  python soft_pseudo.py --captions cast/mevid_captions_mw3.jsonl \\
      --pseudo cast/pseudo_mevid_mw3_th04.json --agree-from cast/mevid_captions_mw3.jsonl \\
      --out cast/soft_mevid_mw3_th04.npz
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_pseudo_clothes import DEFAULT_FIELDS, jaccard, tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captions', required=True)
    ap.add_argument('--pseudo', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--tau', type=float, default=0.1, help='softmax temperature for Q')
    ap.add_argument('--fields', default=','.join(DEFAULT_FIELDS))
    ap.add_argument('--agree-from', default='',
                    help='multi-window captions jsonl; scales w by cluster-mean '
                         'clothes-field vote agreement (the C3 channel)')
    ap.add_argument('--w-mode', default='margin', choices=['margin', 'coherence'],
                    help='confidence definition. margin (legacy): top1-top2 of Q '
                         '-- flawed under over-segmentation, a pure cluster split '
                         'from its wording-twin gets margin~0 and is silenced '
                         'although its adversarial anchor is correct. coherence '
                         '(2026-08-04 fix): members mean pairwise caption Jaccard '
                         '-- measures whether the cluster agrees with itself, '
                         'indifferent to twins.')
    args = ap.parse_args()
    fields = tuple(args.fields.split(','))

    pmap = json.load(open(args.pseudo))['map']
    C = json.load(open(args.pseudo))['num_clothes']

    per_cluster = defaultdict(list)   # cluster -> [token set]
    for line in open(args.captions):
        r = json.loads(line)
        if r.get('split', args.split) != args.split or not r.get('caption'):
            continue
        c = pmap.get(str(r['tracklet']))
        if c is None:
            continue
        per_cluster[c].append(tokens(r['caption'], fields))

    union = [set() for _ in range(C)]
    for c, toks in per_cluster.items():
        for t in toks:
            union[c] |= t

    S = np.eye(C, dtype=np.float32)
    for a in range(C):
        for b in range(a + 1, C):
            s = jaccard(union[a], union[b])
            S[a, b] = S[b, a] = s

    # Q: average over a cluster's members of softmax(similarity-to-every-cluster / tau).
    # A tight, distinctive cluster concentrates on itself; a loose one spreads onto
    # the outfits it is genuinely confusable with -- which is what the adversarial
    # target should say instead of a confident lie.
    Q = np.zeros((C, C), dtype=np.float32)
    w = np.zeros(C, dtype=np.float32)
    for c in range(C):
        members = per_cluster.get(c, [])
        if not members:
            Q[c, c] = 1.0
            w[c] = 0.0
            continue
        rows = []
        for t in members:
            sim = np.array([jaccard(t, union[k]) for k in range(C)], dtype=np.float32)
            e = np.exp((sim - sim.max()) / args.tau)
            rows.append(e / e.sum())
        R = np.stack(rows)
        Q[c] = R.mean(0)
        if args.w_mode == 'coherence':
            # Reliability = do the members' captions agree with EACH OTHER.
            # A twin cluster (one outfit split in two by wording) stays high;
            # only genuinely incoherent clusters get down-weighted. Singletons
            # carry no internal evidence -> sentinel, filled with global mean.
            if len(members) < 2:
                w[c] = -1.0
            else:
                acc, n = 0.0, 0
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        acc += jaccard(members[i], members[j])
                        n += 1
                w[c] = acc / n
        else:
            top2 = np.sort(Q[c])[-2:]
            w[c] = float(np.clip(top2[1] - top2[0], 0.0, 1.0))

    if args.w_mode == 'coherence':
        known = w[w >= 0.0]
        fill = float(known.mean()) if known.size else 0.5
        n_single = int((w < 0.0).sum())
        w[w < 0.0] = fill
        print('coherence w: %d singleton clusters filled with global mean %.3f'
              % (n_single, fill))

    if args.agree_from:
        # C3: per-tracklet mean vote share over clothes.* fields -> cluster mean -> w scale
        agr = {}
        for line in open(args.agree_from):
            r = json.loads(line)
            if r.get('split', args.split) != args.split:
                continue
            a = r.get('agree') or {}
            vals = [v for k2, v in a.items() if k2.startswith('clothes.')]
            if vals:
                agr[str(r['tracklet'])] = float(sum(vals) / len(vals))
        cluster_agree = defaultdict(list)
        for t, c in pmap.items():
            if str(t) in agr:
                cluster_agree[c].append(agr[str(t)])
        scaled = 0
        for c in range(C):
            if cluster_agree.get(c):
                w[c] *= sum(cluster_agree[c]) / len(cluster_agree[c])
                scaled += 1
        print('C3 agree scaling: %d/%d clusters scaled, tracklets with agree: %d'
              % (scaled, C, len(agr)))

    drift = (1.0 - S).astype(np.float32)
    np.fill_diagonal(drift, 0.0)

    np.savez(args.out, S=S, Q=Q, w=w, drift=drift, num_clothes=C)
    print('clusters: %d' % C)
    print('confidence w [%s]: mean %.3f  std %.3f  min %.3f  max %.3f  (<0.3: %d clusters)'
          % (args.w_mode, w.mean(), w.std(), w.min(), w.max(), int((w < 0.3).sum())))
    print('Q self-mass: mean %.3f  (1.0 would mean hard labels)'
          % float(np.mean([Q[c, c] for c in range(C)])))
    print('drift among distinct clusters: mean %.3f' % drift[drift > 0].mean())
    print('wrote %s' % args.out)


if __name__ == '__main__':
    main()
