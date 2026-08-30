# Added by the CAST authors (2026). Not part of upstream Simple-CCReID.
# Distributed under the Apache License 2.0, alongside the Simple-CCReID
# codebase (Gu et al., CVPR 2022) that this file plugs into.
"""CAST losses for learning clothes-invariance from noisy caption supervision.

Two of the three CAST mechanisms live here (the third, cross-window caption
voting, runs offline and reaches training through the confidence vector `w`):

  SoftClothesAdversarialLoss  -- CAL's adversarial target, but built from the
      graded caption evidence instead of a hard pseudo one-hot, and down-weighted
      where that evidence is thin.
  DriftAlignLoss              -- pulls together tracklets of one identity that the
      captions say are wearing different outfits, weighted by how far the outfit
      description actually drifted. Supplies the cross-clothes positive pairs that
      GT-free training otherwise cannot identify.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftClothesAdversarialLoss(nn.Module):
    """Confidence-calibrated soft variant of CAL's clothes-based adversarial loss.

    CAL builds its target as
        mask = (1-eps) * onehot(pseudo_label) + eps/|P| * positive_mask
    which asserts the pseudo label is correct. Caption clusters are only ~84% pure,
    so that one-hot is a confident lie on roughly one tracklet in six.

    Here the one-hot is replaced by Q[target] renormalised over the identity's own
    outfits -- the evidence the clustering actually had -- and each sample's loss is
    scaled by the cluster's confidence w, so loose or ambiguous clusters stop
    steering the adversary. w is normalised to mean 1 within the batch, keeping the
    loss on the same scale as CAL's for a like-for-like comparison.

    Args:
        Q: (C, C) soft target rows per pseudo-outfit.
        w: (C,) per-cluster confidence in [0, 1].
        scale, epsilon: as in CAL.
        use_soft / use_conf: ablation switches (both off == CAL exactly).
    """

    def __init__(self, Q, w, scale=16, epsilon=0.1, use_soft=True, use_conf=True):
        super().__init__()
        self.register_buffer('Q', torch.as_tensor(Q).float())
        self.register_buffer('w', torch.as_tensor(w).float())
        self.scale = scale
        self.epsilon = epsilon
        self.use_soft = use_soft
        self.use_conf = use_conf

    def forward(self, inputs, targets, positive_mask):
        inputs = self.scale * inputs
        negative_mask = 1 - positive_mask

        exp_logits = torch.exp(inputs)
        log_sum_exp_pos_and_all_neg = torch.log(
            (exp_logits * negative_mask).sum(1, keepdim=True) + exp_logits)
        log_prob = inputs - log_sum_exp_pos_and_all_neg

        if self.use_soft:
            # keep the soft mass on this identity's own outfits; leaking onto other
            # identities' clusters would pull the feature toward the wrong person
            anchor = self.Q[targets] * positive_mask
            anchor = anchor / anchor.sum(1, keepdim=True).clamp(min=1e-8)
        else:
            anchor = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1.0)

        mask = ((1 - self.epsilon) * anchor
                + self.epsilon / positive_mask.sum(1, keepdim=True) * positive_mask)
        per_sample = (-mask * log_prob).sum(1)

        if self.use_conf:
            conf = self.w[targets]
            conf = conf / conf.mean().clamp(min=1e-8)
            # importance-weight clipping: on MEVID 84% of clusters sit below 0.3
            # confidence, so batch-normalisation can hand a single confident sample
            # a ~9x weight and destabilise the adversary; cap the ratio at 4x
            conf = conf.clamp(max=4.0)
            return (per_sample * conf).mean()
        return per_sample.mean()


class DriftAlignLoss(nn.Module):
    """Drift-weighted alignment across caption-detected outfit changes.

    Clothes-invariance is decided on pairs showing one person in two outfits, but
    without clothes annotations those pairs cannot be named. Caption drift names
    them: two tracklets of one identity whose outfit descriptions diverge are, with
    high probability, a change of clothes -- and the size of the divergence says how
    hard the pair is. This loss pulls exactly those pairs together, weighted by
    drift, so the pressure concentrates where clothes actually changed rather than
    being spread over every positive pair as the triplet term does.

    Identity-channel persistence (gender/build agree ~98%/85% across a person's
    tracklets) is what licenses reading a clothes-channel change as a real outfit
    change rather than caption noise; that check is done offline when drift is built.

    Args:
        drift: (C, C) 1 - caption similarity between pseudo-outfits, zero diagonal.
    """

    def __init__(self, drift):
        super().__init__()
        self.register_buffer('drift', torch.as_tensor(drift).float())

    def forward(self, features, pids, clothes_ids):
        f = F.normalize(features, p=2, dim=1)
        same_pid = pids.unsqueeze(0) == pids.unsqueeze(1)
        diff_outfit = clothes_ids.unsqueeze(0) != clothes_ids.unsqueeze(1)
        pair = same_pid & diff_outfit
        pair.fill_diagonal_(False)
        if not pair.any():
            return features.sum() * 0.0

        weight = self.drift[clothes_ids][:, clothes_ids] * pair
        cos = f @ f.t()
        loss = ((1.0 - cos) * weight).sum() / weight.sum().clamp(min=1e-8)
        return loss


def load_cast_supervision(npz_path, device):
    z = np.load(npz_path)
    return (torch.tensor(z['Q']).to(device),
            torch.tensor(z['w']).to(device),
            torch.tensor(z['drift']).to(device))


class CaptionDirSuppress(nn.Module):
    """Caption-directed clothes suppression (mechanism T, 2026-08-13).

    The controlled study showed classifier-based clothes adversaries damage
    strong pretrained backbones (GT supervision: all-protocol mAP drop). This
    is the gentle replacement: each pseudo-outfit cluster's clothes caption is
    embedded once by the FROZEN CLIP text tower into the joint 512-d space,
    and training presses the visual feature out of that one direction --
    cos^2(f, t_c) -- instead of fighting a trained classifier.
    Coherence-calibrated: clusters whose captions do not agree with themselves
    (low omega) exert little pressure.

    Args:
        T: (C, D) L2-normalised clothes-text anchors per pseudo cluster.
        w: (C,)  coherence weights (omega), in [0, 1].
    """

    def __init__(self, T, w):
        super().__init__()
        self.register_buffer('T', torch.as_tensor(T).float())
        self.register_buffer('w', torch.as_tensor(w).float())

    def forward(self, features, clothes_ids):
        f = F.normalize(features, p=2, dim=1)
        t = self.T[clothes_ids]
        cos = (f * t).sum(1)
        return (self.w[clothes_ids] * cos.pow(2)).mean()


class OutfitPrototypeAlign(nn.Module):
    """Drift-weighted cross-outfit prototype alignment (DCPA, 2026-08-13).

    An EMA prototype is kept for every (identity, pseudo-outfit) pair that has
    ever appeared. Each sample is pulled toward the prototypes of its OWN
    identity's other outfits, weighted by caption drift, so the invariance
    signal is dense: it covers outfits absent from the current batch, which
    the in-batch pair loss structurally cannot.

    Args:
        drift: (C, C) caption-drift matrix, zero diagonal.
        num_pids: bank rows (train identities).
        dim: feature dim.
        momentum: EMA momentum for prototype updates.
    """

    def __init__(self, drift, num_pids, dim=512, momentum=0.9):
        super().__init__()
        d = torch.as_tensor(drift).float()
        C = d.shape[0]
        self.momentum = momentum
        self.register_buffer('drift', d)
        self.register_buffer('proto', torch.zeros(num_pids, C, dim))
        self.register_buffer('cnt', torch.zeros(num_pids, C))

    @torch.no_grad()
    def _update(self, f, pids, cids):
        fn = F.normalize(f.detach(), p=2, dim=1)
        for p, c in {(int(p), int(c)) for p, c in zip(pids.tolist(), cids.tolist())}:
            m = (pids == p) & (cids == c)
            mean = fn[m].mean(0)
            if self.cnt[p, c] == 0:
                self.proto[p, c] = mean
            else:
                self.proto[p, c] = self.momentum * self.proto[p, c] + (1 - self.momentum) * mean
            self.proto[p, c] = F.normalize(self.proto[p, c], p=2, dim=0)
            self.cnt[p, c] += 1

    def forward(self, features, pids, clothes_ids):
        fn = F.normalize(features, p=2, dim=1)
        pb = self.proto[pids]                                   # (B, C, D)
        seen = self.cnt[pids] > 0                               # (B, C)
        C = self.drift.shape[0]
        other = torch.ones_like(seen)
        other[torch.arange(fn.shape[0], device=fn.device), clothes_ids] = False
        w = self.drift[clothes_ids] * (seen & other).float()    # (B, C)
        cos = torch.einsum('bd,bcd->bc', fn, pb)
        loss = ((1.0 - cos) * w).sum() / w.sum().clamp(min=1e-8)
        self._update(features, pids, clothes_ids)
        return loss
