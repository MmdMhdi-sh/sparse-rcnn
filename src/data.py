"""Synthetic shapes dataset: 128x128 black images with 1-3 coloured circles, squares and triangles.

Boxes are stored as pixel corners [x1, y1, x2, y2]. The Dataset returns them as
normalised centre boxes [cx, cy, w, h] in [0, 1], which is what the model predicts.
"""
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import DataConfig


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def box_iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def visible_fraction(box, others):
    """Upper bound on the fraction of `box` covered by `others` (overlaps are summed), as 1 - that."""
    area = (box[2] - box[0]) * (box[3] - box[1])
    covered = sum(max(0, min(box[2], o[2]) - max(box[0], o[0])) *
                  max(0, min(box[3], o[3]) - max(box[1], o[1])) for o in others)
    return 1 - min(covered, area) / area


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _sample_box(rng, shape_id, size):
    if shape_id == 0:                                   # circle, radius 10-24
        r = int(rng.integers(10, 25))
        cx, cy = int(rng.integers(r, size - r)), int(rng.integers(r, size - r))
        return [cx - r, cy - r, cx + r, cy + r]
    if shape_id == 1:                                   # square, side 10-39
        w = int(rng.integers(10, 40))
        x, y = int(rng.integers(0, size - w)), int(rng.integers(0, size - w))
        return [x, y, x + w, y + w]
    w, h = int(rng.integers(15, 40)), int(rng.integers(15, 40))   # triangle, 15-39 x 15-39
    x, y = int(rng.integers(0, size - w)), int(rng.integers(0, size - h))
    return [x, y, x + w, y + h]


def _draw(img, shape_id, box, color):
    x1, y1, x2, y2 = box
    if shape_id == 0:
        r = (x2 - x1) // 2
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
    elif shape_id == 1:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    else:
        pts = np.array([[x1, y2], [(x1 + x2) // 2, y1], [x2, y2]], np.int32)
        cv2.fillPoly(img, [pts], color)


def create_image(rng, cfg: DataConfig):
    size = cfg.img_size
    img = np.zeros((size, size, 3), dtype=np.uint8)
    boxes, labels = [], []
    for _ in range(int(rng.integers(1, cfg.max_shapes + 1))):
        for _ in range(50):                             # retry limit; the shape is skipped if all fail
            shape_id = int(rng.integers(0, 3))
            box = _sample_box(rng, shape_id, size)
            if any(box_iou(box, b) > cfg.iou_threshold for b in boxes):
                continue
            # The new shape is drawn on top, so check that every *earlier* shape stays visible
            # (the course version only checked the new shape, which is never occluded).
            if any(visible_fraction(b, [box] + [o for o in boxes if o is not b]) < cfg.visibility_threshold
                   for b in boxes):
                continue
            color = rng.integers(cfg.min_color, 256, size=3).tolist()
            _draw(img, shape_id, box, color)
            boxes.append(box)
            labels.append(shape_id)
            break
    return img, boxes, labels


def generate(n, seed, cfg: DataConfig):
    """n images from one RNG stream, so the first k images do not depend on n."""
    rng = np.random.default_rng(seed)
    images, targets = [], []
    for _ in range(n):
        img, boxes, labels = create_image(rng, cfg)
        images.append(img)
        targets.append({"boxes": boxes, "labels": labels})
    return np.stack(images), targets


def make_splits(cfg: DataConfig):
    """Validation = the first n_val pool images (identical for every pool size); train = the rest.
    Test = a separate set generated from its own seed, used only for the final metrics."""
    x_pool, y_pool = generate(cfg.n_pool, cfg.pool_seed, cfg)
    x_test, y_test = generate(cfg.n_test, cfg.test_seed, cfg)
    return {"train": (x_pool[cfg.n_val:], y_pool[cfg.n_val:]),
            "val": (x_pool[:cfg.n_val], y_pool[:cfg.n_val]),
            "test": (x_test, y_test)}


def split_stats(splits):
    out = {}
    for name, (x, y) in splits.items():
        labels = [l for t in y for l in t["labels"]]
        out[name] = {"images": len(x), "objects": len(labels),
                     "per_class": [labels.count(c) for c in range(3)]}
    return out


# ---------------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------------
class ShapesDataset(Dataset):
    def __init__(self, images, targets, img_size):
        self.images, self.targets, self.size = images, targets, img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float() / 255.0
        b = torch.tensor(self.targets[idx]["boxes"], dtype=torch.float32).reshape(-1, 4) / self.size
        boxes = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                             b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], dim=1)
        labels = torch.tensor(self.targets[idx]["labels"], dtype=torch.int64)
        return img, {"boxes": boxes, "labels": labels}


def collate_fn(batch):
    """Images are stacked; targets stay a list because each image has a different number of objects."""
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def make_loader(split, cfg: DataConfig, shuffle, seed=0):
    x, y = split
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ShapesDataset(x, y, cfg.img_size), batch_size=cfg.batch_size,
                      shuffle=shuffle, collate_fn=collate_fn, generator=g)
