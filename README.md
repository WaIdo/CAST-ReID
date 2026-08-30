# CAST

Reference implementation of CAST for clothes-changing video person
re-identification. The corresponding manuscript is under review.

> **Scope.** Review-stage release: the implementation, together with the
> caption, pseudo-label and weight artefacts behind the reported models, so the
> results can be checked and re-derived. The full release — all training and
> ablation scripts, configurations, run logs and checkpoints — **will be public
> upon acceptance** ([details](#release-plan)).

## Contents

```
cast/
  caption_mevid.py          Offline captioning with a frozen MLLM
  build_pseudo_clothes.py   Builds the pseudo-label file from captions
  soft_pseudo.py            Precomputes the weight file used in training
  soup_ckpt.py              Uniform parameter average over checkpoints
  bn_recalib.py             BatchNorm re-estimation (ResNet-50 only)
  paired_test.py            Per-query scores and paired bootstrap test
losses/cast_losses.py       Training objectives
configs/                    Configs of the deployed MEVID and CCVID models
data/                       Captions, pseudo labels, precomputed weights
docs/train_integration.patch  Hunks applied to the Simple-CCReID baseline
```

## Building the supervision artefacts

The captions are included, so these two steps run on CPU in minutes, with no
dataset download, no GPU and no MLLM:

```bash
python cast/build_pseudo_clothes.py \
    --captions data/mevid_captions_train.jsonl \
    --th 0.4 --out /tmp/pseudo_mevid_th04.json

python cast/soft_pseudo.py \
    --captions data/mevid_captions_train.jsonl \
    --pseudo /tmp/pseudo_mevid_th04.json \
    --w-mode coherence \
    --out /tmp/soft_mevid_th04_cohw.npz
```

Both are deterministic and reproduce the shipped artefacts in `data/` exactly.
Use `--th 0.7` for CCVID.

To regenerate captions from raw video, `cast/caption_mevid.py` runs a frozen
Qwen2.5-VL-7B-Instruct over the tracklets (GPU and video data required).

## Training and evaluation

Training builds on the public [Simple-CCReID](https://github.com/guxinqian/Simple-CCReID)
codebase (CAL, CVPR 2022). Place `losses/cast_losses.py` and `configs/` into
that checkout, apply `docs/train_integration.patch`, then, for each of three
seeds:

```bash
CAST_PSEUDO=data/pseudo_mevid_th04.json \
CAST_SOFT=data/soft_mevid_th04_cohw.npz \
CAST_C2_ONLY=1 CAST_ALIGN_W=0.2 CLIP_UNFREEZE=1 \
torchrun --nproc_per_node=1 main.py \
    --dataset mevid --cfg configs/mevid_clipft_c2only30.yaml --root $DATA_ROOT \
    --output ./logs_cast --gpu 0
```

Then average the three checkpoints and evaluate:

```bash
python cast/soup_ckpt.py cast_soup.pth s1.pth.tar s2.pth.tar s3.pth.tar
torchrun --nproc_per_node=1 main.py --dataset mevid \
    --cfg configs/mevid_clipft_c2only30.yaml --root $DATA_ROOT \
    --eval --resume cast_soup.pth --gpu 0
```

Every reported number comes from the fixed final epoch, with no checkpoint
selection on the test set. Runs used a single NVIDIA RTX 5090 (32 GB).

### Environment variables

Paths are set by environment variable; nothing is hard-coded.

| Variable | Meaning | Default |
|---|---|---|
| `DATA_ROOT` | Parent of `MEVID/` and `CCVID/` | `./datasets` |
| `CCREID_ROOT` | Simple-CCReID checkout | `./Simple-CCReID` |
| `QWEN_VL` | Qwen2.5-VL-7B-Instruct weights | `./pretrained/Qwen2.5-VL-7B-Instruct` |
| `CAST_PSEUDO` / `CAST_SOFT` | Pseudo labels / precomputed weights | — |
| `CAST_ALIGN_W` | Weight of the alignment term | `0.5` |
| `CAST_C2_ONLY` | `1` = deployed configuration | `0` |

## Setup

```
Python 3.8+, PyTorch 2.x, yacs, numpy, scipy
transformers + qwen-vl-utils    (only to regenerate captions)
```

Experiments ran on Python 3.11 / PyTorch 2.6.

## Data

Datasets are **not** redistributed. Obtain
[MEVID](https://github.com/Kitware/MEVID) and
[CCVID](https://github.com/guxinqian/Simple-CCReID) from their official sources,
under their own licenses and access terms.

`data/*.jsonl` holds machine-generated captions for the training tracklets of
those benchmarks, released as derived text annotations so that the supervision
is auditable. Each record carries the benchmark's own tracklet, identity and
camera indices plus one JSON caption over a fixed schema. Values are coarse
attribute words produced by a frozen MLLM, and many are `unknown`. No imagery is
redistributed, and users remain bound by the source datasets' terms.

## Release plan

| | This release (review) | Full release (upon acceptance) |
|---|---|---|
| Implementation | ✅ | ✅ |
| Captions, pseudo labels, weights | ✅ deployed | ✅ all variants |
| Model configurations | ✅ deployed | ✅ all 35 |
| Evaluation + significance test | ✅ | ✅ |
| Baseline integration | ✅ as a patch | ✅ full framework |
| Training / ablation scripts | — | ✅ 54 |
| Per-run logs for every reported number | — | ✅ 156 runs |
| Figure code and per-query data | — | ✅ |
| Trained checkpoints | — | ✅ |

## License and attribution

Apache License 2.0 (see [LICENSE](LICENSE)).

Built on [Simple-CCReID](https://github.com/guxinqian/Simple-CCReID) by Gu et
al. ("Clothes-Changing Person Re-identification with RGB Modality Only", CVPR
2022), Apache 2.0; `docs/train_integration.patch` records our modifications.
Captions are generated with
[Qwen2.5-VL-7B-Instruct](https://github.com/QwenLM/Qwen2.5-VL) (Apache 2.0); the
backbone is initialised from [CLIP](https://github.com/openai/CLIP) ViT-B/16
(MIT). Neither model's weights are redistributed here.
