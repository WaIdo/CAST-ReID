#!/usr/bin/env python3
"""CAST: caption MEVID training tracklets with the v2 dual-channel schema. Resumable JSONL.

MEVID keeps every frame of a person in one flat directory (bbox_train/0103/), so a
tracklet cannot be identified by its folder the way CCVID's can. The frame name
encodes it instead: 0103O020C424T000F00000.jpg -> pid 0103, outfit 020, cam 424,
tracklet 000. We key on everything before the F field, which is unique per tracklet.

Set MEVID_SPLIT=test to caption the test split instead (not needed for pseudo labels,
which only feed the training-time clothes classifier).
"""
import json
import os
import re
import time

import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

ROOT = os.path.join(os.environ.get('DATA_ROOT', './datasets'), 'MEVID')
ANNO = os.path.join(ROOT, 'mevid-v1-annotation-data')
MODEL = os.environ.get('QWEN_VL', './pretrained/Qwen2.5-VL-7B-Instruct')
SPLIT = os.environ.get('MEVID_SPLIT', 'train')
OUT = os.path.join(os.environ.get('CAST_OUT', '.'), 'mevid_captions_%s.jsonl' % SPLIT)

PROMPT = (
    'You are annotating a short surveillance video clip of ONE pedestrian.\n'
    'Return STRICT JSON only, no markdown, with exactly this schema:\n'
    '{"non_clothes": {"gender": str, "age_band": str, "build": str, "hair": str}, '
    '"clothes": {"upper": str, "lower": str, "shoes": str, "accessories_carried": str, '
    '"dominant_colors": str}, '
    '"motion": {"gait_speed": str, "posture": str}}\n'
    'Rules: describe ONLY what is clearly visible; write "unknown" if not visible; '
    'non_clothes fields must contain NO clothing or accessory information; '
    'be concise (<=6 words per field).'
)


def all_tracklets():
    """Return (key, pid, outfit, camid, frame_paths) per tracklet, annotation order."""
    names = [l.strip() for l in open(os.path.join(ANNO, '%s_name.txt' % SPLIT))]
    track = np.loadtxt(os.path.join(ANNO, 'track_%s_info.txt' % SPLIT)).astype(int)
    subdir = 'bbox_%s' % SPLIT
    out = []
    for row in track:
        if row[1] < row[0]:  # one train row has end < start (empty tracklet)
            continue
        frames = names[row[0]:row[1] + 1]
        key = frames[0].split('F')[0]
        paths = [os.path.join(ROOT, subdir, f[:4], f) for f in frames]
        out.append((key, int(row[2]), int(row[3]), int(row[4]), paths))
    return out


def pick_frames(paths, n=6):
    if not paths:
        return None
    idx = [int(i) for i in torch.linspace(0, len(paths) - 1, min(n, len(paths))).tolist()]
    return [paths[i] for i in idx]


def main():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                done.add(json.loads(line)['tracklet'])
            except Exception:
                pass
    todo = [t for t in all_tracklets() if t[0] not in done]
    print('todo %d (done %d)' % (len(todo), len(done)), flush=True)
    if not todo:
        return

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map='cuda')
    proc = AutoProcessor.from_pretrained(MODEL)
    f = open(OUT, 'a')
    t0, n_ok = time.time(), 0
    for i, (key, pid, outfit, camid, paths) in enumerate(todo):
        fr = pick_frames(paths)
        if fr is None:
            continue
        msgs = [{'role': 'user', 'content': [
            {'type': 'video', 'video': fr, 'resized_height': 448, 'resized_width': 224},
            {'type': 'text', 'text': PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inputs = proc(text=[text], images=imgs, videos=vids, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        ans = proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        m = re.search(r'\{.*\}', ans, re.S)
        cap = None
        if m:
            try:
                cap = json.loads(m.group(0))
            except Exception:
                pass
        if cap:
            n_ok += 1
        f.write(json.dumps({'tracklet': key, 'pid': pid, 'gt_clothes': outfit,
                            'camid': camid, 'split': SPLIT, 'caption': cap,
                            'raw': None if cap else ans[:400]}, ensure_ascii=False) + '\n')
        if (i + 1) % 20 == 0:
            f.flush()
            r = (i + 1) / (time.time() - t0)
            print('%d/%d ok=%d %.2f trk/s ETA %.0fmin'
                  % (i + 1, len(todo), n_ok, r, ((len(todo) - i - 1) / r) / 60), flush=True)
    f.close()
    print('DONE ok=%d/%d' % (n_ok, len(todo)), flush=True)


if __name__ == '__main__':
    main()
