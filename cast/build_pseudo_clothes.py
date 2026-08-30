#!/usr/bin/env python3
"""CAST: derive pseudo clothes labels from MLLM tracklet captions.

The clothes classifier CAL trains against only needs to know which tracklets of a
person show the same outfit. We recover that partition from the caption's clothes
channel: tokenise the outfit fields, then greedily group a person's tracklets whose
token sets overlap above a Jaccard threshold. Identity fields are deliberately
unused -- they persist across outfits and would merge everything.

Shoes are excluded by default: in the persistence study they separated same/different
outfits by only +0.04 Jaccard (upper +0.26, colors +0.19, lower +0.12), i.e. noise.

Usage:
  python build_pseudo_clothes.py --captions cast/ccvid_captions.jsonl \\
      --out cast/pseudo_ccvid_th07.json --th 0.7 [--split train]
"""
import argparse
import json
import re
from collections import defaultdict

STOP = {'and', 'with', 'unknown', 'none', 'a', 'an', 'the', 'or', 'plain'}
DEFAULT_FIELDS = ('upper', 'lower', 'dominant_colors')


def tokens(caption, fields):
    out = set()
    cl = (caption or {}).get('clothes', {}) or {}
    for f in fields:
        for w in re.split(r'[^a-z0-9]+', str(cl.get(f, '')).lower()):
            if w and w not in STOP:
                out.add(w)
    return out


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cluster_one_person(items, th):
    """Greedy agglomeration: a tracklet joins the first cluster whose centroid
    (token set intersection-over-union against the cluster's union) clears `th`,
    otherwise it opens a new cluster. Order follows tracklet name, so the result
    is deterministic."""
    clusters = []  # list of (token_union, [indices])
    for idx, toks in items:
        best, best_s = None, th
        for c in clusters:
            s = jaccard(toks, c[0])
            if s >= best_s:
                best, best_s = c, s
        if best is None:
            clusters.append([set(toks), [idx]])
        else:
            best[0] |= toks
            best[1].append(idx)
    return [c[1] for c in clusters]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captions', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--th', type=float, default=0.7)
    ap.add_argument('--split', default='train')
    ap.add_argument('--fields', default=','.join(DEFAULT_FIELDS))
    args = ap.parse_args()
    fields = tuple(args.fields.split(','))

    recs = []
    for line in open(args.captions):
        r = json.loads(line)
        if r.get('split') != args.split or not r.get('caption'):
            continue
        recs.append(r)
    print('tracklets in split %s: %d' % (args.split, len(recs)))

    by_pid = defaultdict(list)
    for r in recs:
        by_pid[str(r['pid'])].append(r)

    mapping, next_id = {}, 0
    gt_pairs = []  # (pseudo_id, gt_clothes) for purity
    n_pseudo_per_pid, n_gt_per_pid = [], []
    for pid in sorted(by_pid):
        rs = sorted(by_pid[pid], key=lambda r: str(r['tracklet']))
        items = [(i, tokens(r['caption'], fields)) for i, r in enumerate(rs)]
        groups = cluster_one_person(items, args.th)
        for g in groups:
            for i in g:
                mapping[rs[i]['tracklet']] = next_id
                gt_pairs.append((next_id, '%s|%s' % (pid, rs[i]['gt_clothes'])))
            next_id += 1
        n_pseudo_per_pid.append(len(groups))
        n_gt_per_pid.append(len(set(str(r['gt_clothes']) for r in rs)))

    # purity: within each pseudo cluster, the share held by its dominant GT outfit
    by_cluster = defaultdict(list)
    for c, gt in gt_pairs:
        by_cluster[c].append(gt)
    purity = sum(max(v.count(x) for x in set(v)) for v in by_cluster.values()) / len(gt_pairs)

    json.dump({'map': mapping, 'num_clothes': next_id}, open(args.out, 'w'))
    print('pseudo clothes ids: %d   (GT outfits: %d)'
          % (next_id, sum(n_gt_per_pid)))
    print('clusters per identity: pseudo %.2f vs GT %.2f'
          % (sum(n_pseudo_per_pid) / len(n_pseudo_per_pid),
             sum(n_gt_per_pid) / len(n_gt_per_pid)))
    print('purity vs GT outfits: %.3f' % purity)
    print('wrote %s' % args.out)


if __name__ == '__main__':
    main()
