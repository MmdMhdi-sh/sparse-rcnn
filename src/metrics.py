"""COCO-style mAP (torchmetrics) for every refinement stage, plus a detection-count check.

Standard mAP ranks *all* detections by score, so no score threshold is applied by default.
The course version kept only scores > 0.3 before computing mAP; that number is reported
separately (`map_thr0.3`) for comparison.
"""
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from config import DEVICE
from model import to_corners

SCORE_THRESHOLD = 0.3     # for visualisation, the count check and the course-style mAP


def postprocess(logits, boxes, img_size, threshold=0.0):
    """Per-image dicts in pixel xyxy: score = highest non-background probability."""
    probs = logits.softmax(-1)[..., :-1]
    scores, labels = probs.max(-1)
    corners = to_corners(boxes) * img_size
    out = []
    for b in range(logits.size(0)):
        keep = scores[b] > threshold
        out.append({"boxes": corners[b][keep].cpu(), "scores": scores[b][keep].cpu(),
                    "labels": labels[b][keep].cpu()})
    return out


def targets_to_pixels(targets, img_size):
    return [{"boxes": (to_corners(t["boxes"]) * img_size).cpu(), "labels": t["labels"].cpu()} for t in targets]


def _summary(metric, class_names):
    r = metric.compute()
    out = {"map": r["map"].item(), "map_50": r["map_50"].item(), "map_75": r["map_75"].item(),
           "mar_100": r["mar_100"].item()}
    if "map_per_class" in r and "classes" in r:
        per_class = dict(zip(r["classes"].flatten().tolist(), r["map_per_class"].flatten().tolist()))
        out["ap_per_class"] = {name: per_class.get(i, float("nan")) for i, name in enumerate(class_names)}
    return out


@torch.no_grad()
def evaluate(model, loader, img_size, class_names, full=True):
    """full=False: final-stage mAP only (used every epoch for checkpoint selection).
    full=True: mAP of every stage, course-style thresholded mAP, and the count check."""
    model.eval()
    n_stages = model.cfg.num_stages
    stages = range(n_stages) if full else [n_stages - 1]
    maps = {s: MeanAveragePrecision(box_format="xyxy", class_metrics=full) for s in stages}
    thr_map = MeanAveragePrecision(box_format="xyxy") if full else None
    n_imgs = exact = over = under = 0
    for images, targets in loader:
        logits, boxes = model(images.to(DEVICE))
        gt = targets_to_pixels(targets, img_size)
        for s in stages:
            maps[s].update(postprocess(logits[s], boxes[s], img_size), gt)
        if full:
            preds = postprocess(logits[-1], boxes[-1], img_size, SCORE_THRESHOLD)
            thr_map.update(preds, gt)
            for p, t in zip(preds, gt):
                d = len(p["labels"]) - len(t["labels"])
                n_imgs += 1
                exact += d == 0
                over += d > 0
                under += d < 0
    if not full:
        return {"map": maps[n_stages - 1].compute()["map"].item()}
    out = _summary(maps[n_stages - 1], class_names)
    out["map_per_stage"] = [_summary(maps[s], class_names)["map"] for s in stages]
    out["map_thr0.3"] = thr_map.compute()["map"].item()
    out["count_exact"] = exact / n_imgs        # share of images with as many confident boxes as objects
    out["count_over"] = over / n_imgs          # duplicates or false positives
    out["count_under"] = under / n_imgs        # missed objects
    return out
