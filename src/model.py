"""A small Sparse R-CNN (Sun et al., 2021) for 128x128 images.

backbone (3 conv blocks, optional FPN) -> 20 learnable proposal boxes + features
-> 3 refinement stages, each: RoIAlign -> dynamic instance interaction -> class / box heads.
Boxes are normalised [cx, cy, w, h] in [0, 1].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from config import ModelConfig


class Backbone(nn.Module):
    """Three conv-BN-ReLU-maxpool blocks: 128 -> 64 -> 32 -> 16 px, ending in 128 channels.
    With FPN, a top-down pathway gives 128-channel maps at 64x64 (P2), 32x32 (P3) and 16x16 (P4)."""

    def __init__(self, out_channels=128, use_fpn=True):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.stage1, self.stage2, self.stage3 = block(3, 32), block(32, 64), block(64, 128)
        self.use_fpn = use_fpn
        if use_fpn:
            self.lateral_p4 = nn.Conv2d(128, out_channels, 1)
            self.lateral_p3 = nn.Conv2d(64, out_channels, 1)
            self.lateral_p2 = nn.Conv2d(32, out_channels, 1)
            self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.smooth_p2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        else:
            assert out_channels == 128, "without FPN the feature map is the 128-channel C4 output"

    def forward(self, x):
        c2 = self.stage1(x)            # [B, 32, 64, 64]
        c3 = self.stage2(c2)           # [B, 64, 32, 32]
        c4 = self.stage3(c3)           # [B, 128, 16, 16]
        if not self.use_fpn:
            return {"p4": c4}
        p4 = self.lateral_p4(c4)
        p3 = self.smooth_p3(F.interpolate(p4, scale_factor=2, mode="nearest") + self.lateral_p3(c3))
        p2 = self.smooth_p2(F.interpolate(p3, scale_factor=2, mode="nearest") + self.lateral_p2(c2))
        return {"p2": p2, "p3": p3, "p4": p4}


class DynamicInteraction(nn.Module):
    """Each proposal feature generates the weights of two 1x1 convolutions that are applied to
    its own RoI features (C -> C/4 -> C), followed by a linear layer over the flattened RoI."""

    def __init__(self, dim=128, roi_size=7):
        super().__init__()
        self.dim, self.mid = dim, dim // 4
        self.param_gen = nn.Linear(dim, 2 * dim * self.mid)
        self.norm1 = nn.GroupNorm(4, self.mid)
        self.norm2 = nn.GroupNorm(4, dim)
        self.out = nn.Linear(roi_size * roi_size * dim, dim)

    def forward(self, pro_feats, roi_feats):
        """pro_feats [M, C], roi_feats [M, S*S, C] -> object features [M, C]."""
        m = pro_feats.size(0)
        params = self.param_gen(pro_feats)
        w1 = params[:, :self.dim * self.mid].view(m, self.dim, self.mid)
        w2 = params[:, self.dim * self.mid:].view(m, self.mid, self.dim)
        x = F.relu(self.norm1(torch.bmm(roi_feats, w1).transpose(1, 2))).transpose(1, 2)
        x = F.relu(self.norm2(torch.bmm(x, w2).transpose(1, 2))).transpose(1, 2)
        return self.out(x.flatten(1))


def to_corners(boxes, eps=1e-4):
    """[cx, cy, w, h] -> [x1, y1, x2, y2] clipped to the image, with x1 < x2 and y1 < y2
    enforced by torch.max (as the assignment asks)."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = torch.clamp(cx - w / 2, 0.0, 1.0)
    y1 = torch.clamp(cy - h / 2, 0.0, 1.0)
    x2 = torch.max(torch.clamp(cx + w / 2, 0.0, 1.0), x1 + eps)
    y2 = torch.max(torch.clamp(cy + h / 2, 0.0, 1.0), y1 + eps)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def to_centers(corners):
    x1, y1, x2, y2 = corners.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


class SparseRCNN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        n, c = cfg.num_proposals, cfg.feature_dim
        self.backbone = Backbone(c, cfg.use_fpn)
        # learnable proposals: boxes start uniformly random in [0, 1]^4 (cx, cy, w, h)
        self.proposal_boxes = nn.Parameter(torch.rand(n, 4))
        self.proposal_feats = nn.Parameter(torch.randn(n, c))
        self.interactions = nn.ModuleList(DynamicInteraction(c, cfg.roi_size) for _ in range(cfg.num_stages))
        self.cls_heads = nn.ModuleList(nn.Linear(c, cfg.num_classes + 1) for _ in range(cfg.num_stages))
        self.box_heads = nn.ModuleList(nn.Linear(c, 4) for _ in range(cfg.num_stages))

    def _levels(self, corners):
        """Assign each box to an FPN level by its normalised area (small -> high-resolution P2)."""
        if not self.cfg.use_fpn:
            return [("p4", torch.ones(corners.shape[:2], dtype=torch.bool, device=corners.device))]
        area = (corners[..., 2] - corners[..., 0]) * (corners[..., 3] - corners[..., 1])
        lo, hi = self.cfg.level_area_thresholds
        return [("p2", area < lo), ("p3", (area >= lo) & (area < hi)), ("p4", area >= hi)]

    def _pool(self, feats, corners):
        """RoIAlign for all B x N boxes at once -> [B*N, S*S, C], in proposal order."""
        b, n, _ = corners.shape
        s, c = self.cfg.roi_size, self.cfg.feature_dim
        out = corners.new_zeros(b, n, c, s, s)
        batch_idx = torch.arange(b, device=corners.device).view(b, 1).expand(b, n)
        for name, mask in self._levels(corners):
            if not mask.any():
                continue
            fmap = feats[name]
            rois = torch.cat([batch_idx[mask].unsqueeze(1).to(corners.dtype),
                              corners[mask] * fmap.shape[-1]], dim=1)    # feature-map pixels
            out[mask] = roi_align(fmap, rois, output_size=s, spatial_scale=1.0, aligned=True)
        return out.flatten(3).transpose(2, 3).reshape(b * n, s * s, c)

    def forward(self, images):
        """Returns per-stage lists of class logits [B, N, K+1] and boxes [B, N, 4] (cx, cy, w, h)."""
        b, n, c = images.size(0), self.cfg.num_proposals, self.cfg.feature_dim
        feats = self.backbone(images)
        boxes = self.proposal_boxes.unsqueeze(0).expand(b, n, 4)
        pro_feats = self.proposal_feats.unsqueeze(0).expand(b, n, c).reshape(b * n, c)
        all_logits, all_boxes = [], []
        for stage in range(self.cfg.num_stages):
            corners = to_corners(boxes)              # keep boxes valid before RoIAlign
            boxes = to_centers(corners)
            obj = self.interactions[stage](pro_feats, self._pool(feats, corners))
            logits = self.cls_heads[stage](obj).view(b, n, -1)
            dx, dy, dw, dh = self.box_heads[stage](obj).view(b, n, 4).unbind(-1)
            m = self.cfg.max_log_scale
            cx, cy, w, h = boxes.unbind(-1)
            boxes = torch.stack([cx + dx * w, cy + dy * h,
                                 w * torch.exp(dw.clamp(-m, m)), h * torch.exp(dh.clamp(-m, m))], dim=-1)
            all_logits.append(logits)
            all_boxes.append(boxes)
            boxes = boxes.detach()                   # as in the paper: no gradient across stages
            pro_feats = obj
        return all_logits, all_boxes


def count_params(module):
    return sum(p.numel() for p in module.parameters())
