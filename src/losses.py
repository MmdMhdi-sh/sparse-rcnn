"""Set prediction loss: Hungarian matching (class + L1 cost), focal classification loss, L1 box loss.

Every stage is supervised (deep supervision) and matched independently.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from config import LossConfig


@torch.no_grad()
def hungarian_match(logits, boxes, targets, cost_class=1.0, cost_bbox=1.0):
    """One-to-one assignment of proposals to ground-truth objects per image.
    Returns a list of (proposal_idx, target_idx) tensor pairs."""
    probs = logits.softmax(-1)
    out = []
    for b, t in enumerate(targets):
        if len(t["labels"]) == 0:
            empty = torch.empty(0, dtype=torch.int64, device=logits.device)
            out.append((empty, empty))
            continue
        cost = -cost_class * probs[b][:, t["labels"]] + cost_bbox * torch.cdist(boxes[b], t["boxes"], p=1)
        rows, cols = linear_sum_assignment(cost.cpu().numpy())
        out.append((torch.as_tensor(rows, device=logits.device), torch.as_tensor(cols, device=logits.device)))
    return out


class SetCriterion(nn.Module):
    def __init__(self, num_classes, cfg: LossConfig):
        super().__init__()
        self.bg = num_classes            # background is the last class index
        self.cfg = cfg

    def stage_loss(self, logits, boxes, targets):
        idx = hungarian_match(logits, boxes, targets, self.cfg.cost_class, self.cfg.cost_bbox)
        target_cls = torch.full(logits.shape[:2], self.bg, dtype=torch.int64, device=logits.device)
        for b, (src, tgt) in enumerate(idx):
            target_cls[b, src] = targets[b]["labels"][tgt]
        # multi-class focal loss over all proposals (background included)
        ce = F.cross_entropy(logits.flatten(0, 1), target_cls.flatten(), reduction="none")
        loss_cls = ((1 - torch.exp(-ce)) ** self.cfg.focal_gamma * ce).mean()
        # L1 box loss on matched proposals only
        src_boxes = torch.cat([boxes[b, s] for b, (s, _) in enumerate(idx)])
        tgt_boxes = torch.cat([targets[b]["boxes"][t] for b, (_, t) in enumerate(idx)])
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes) if len(src_boxes) else boxes.sum() * 0
        return loss_cls, loss_bbox

    def forward(self, outputs, targets):
        all_logits, all_boxes = outputs
        total, logs = 0.0, {}
        for s, (logits, boxes) in enumerate(zip(all_logits, all_boxes)):
            loss_cls, loss_bbox = self.stage_loss(logits, boxes, targets)
            total = total + self.cfg.weight_cls * loss_cls + self.cfg.weight_bbox * loss_bbox
            logs[f"cls_s{s + 1}"] = loss_cls.item()
            logs[f"bbox_s{s + 1}"] = loss_bbox.item()
        logs["total"] = total.item()
        return total, logs
