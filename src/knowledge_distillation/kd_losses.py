import math

import torch
import torch.nn.functional as F


# ---------- Pairing by spatial size (robust to different level sets) ----------
def _feat_hw(x):  # H, W
    return x.shape[-2], x.shape[-1]


def _size_area(x):
    return x.shape[-2] * x.shape[-1]


def _log2_dist(hw_a, hw_b):
    ah, aw = hw_a
    bh, bw = hw_b
    return abs(math.log2(max(ah, 1)) - math.log2(max(bh, 1))) + abs(
        math.log2(max(aw, 1)) - math.log2(max(bw, 1))
    )


def _dict_to_sorted_list(feat_dict: dict):
    items = list(feat_dict.items())
    items.sort(key=lambda kv: kv[1].shape[-2] * kv[1].shape[-1], reverse=True)
    return [t for _, t in items]


def pair_feats_by_size(student_dict: dict, teacher_dict: dict, max_logdist=1.1):
    """
    Greedily pair teacher levels with closest student level by spatial size.
    Returns two lists (student_list, teacher_list) of equal length.
    """
    s_items = list(student_dict.items())
    t_items = list(teacher_dict.items())
    s_items.sort(key=lambda kv: _size_area(kv[1]), reverse=True)
    t_items.sort(key=lambda kv: _size_area(kv[1]), reverse=True)

    used_s = set()
    student_list, teacher_list = [], []
    for tk, tv in t_items:
        t_hw = _feat_hw(tv)
        best_j, best_d = None, 1e9
        for j, (sk, sv) in enumerate(s_items):
            if j in used_s:
                continue
            d = _log2_dist(t_hw, _feat_hw(sv))
            if d < best_d:
                best_d, best_j = d, j
        if best_j is not None and best_d <= max_logdist:
            used_s.add(best_j)
            student_list.append(s_items[best_j][1])
            teacher_list.append(tv)
    return student_list, teacher_list


# ---------- KD losses (feature & RPN) ------------------------------------------
def mse_feature_loss_from_lists(student_list, teacher_list, normalize=True):
    L = min(len(student_list), len(teacher_list))
    loss = 0.0
    for i in range(L):
        s, t = student_list[i], teacher_list[i].detach()
        if s.shape[-2:] != t.shape[-2:]:
            s = F.interpolate(
                s, size=t.shape[-2:], mode="bilinear", align_corners=False
            )
        if normalize:
            s = F.normalize(s, p=2, dim=1)
            t = F.normalize(t, p=2, dim=1)
        loss = loss + F.mse_loss(s, t)
    return loss


def pair_tensors_by_size(student_list, teacher_list, max_logdist=1.1):
    s_items = list(enumerate(student_list))
    t_items = list(enumerate(teacher_list))
    s_items.sort(key=lambda it: _size_area(it[1]), reverse=True)
    t_items.sort(key=lambda it: _size_area(it[1]), reverse=True)

    used_s = set()
    pairs = []
    for _, t in t_items:
        best_j, best_d = None, 1e9
        for j, s in s_items:
            if j in used_s:
                continue
            d = _log2_dist(_feat_hw(s), _feat_hw(t))
            if d < best_d:
                best_d, best_j = d, j
        if best_j is not None and best_d <= max_logdist:
            used_s.add(best_j)
            pairs.append((s, t))
    return pairs


def rpn_objectness_distill(
    student_logits_list, teacher_logits_list, T=3.0, max_logdist=1.1
):
    """
    Robust RPN KD:
      - pair by spatial size
      - resize student to teacher HxW
      - collapse anchors by mean across channel dim
      - BCEWithLogits(student/T, sigmoid(teacher/T)) * T^2
    """
    device = None
    if len(teacher_logits_list):
        device = teacher_logits_list[0].device
    elif len(student_logits_list):
        device = student_logits_list[0].device
    else:
        device = "cpu"

    pairs = pair_tensors_by_size(
        student_logits_list, teacher_logits_list, max_logdist=max_logdist
    )
    if len(pairs) == 0:
        return torch.tensor(0.0, device=device)

    loss = 0.0
    for s, t in pairs:
        # s,t: (N, A, H, W) or (N, A*1, H, W)
        s_c = s.mean(dim=1, keepdim=True)
        t_c = t.mean(dim=1, keepdim=True)
        if s_c.shape[-2:] != t_c.shape[-2:]:
            s_c = F.interpolate(
                s_c, size=t_c.shape[-2:], mode="bilinear", align_corners=False
            )
        loss = loss + F.binary_cross_entropy_with_logits(
            s_c / T, torch.sigmoid(t_c / T)
        ) * (T * T)
    return loss
