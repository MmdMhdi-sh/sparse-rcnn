"""Training loop with validation-mAP checkpoint selection, test evaluation and plots."""
import copy
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import CLASS_NAMES, DEVICE, ExperimentConfig
from data import make_loader
from losses import SetCriterion
from metrics import SCORE_THRESHOLD, evaluate, postprocess
from model import SparseRCNN, to_corners

COLORS = ("#e15759", "#4e79a7", "#59a14f")       # circle, square, triangle


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # also seeds CUDA


def fit(exp: ExperimentConfig, splits, verbose=True):
    set_seed(exp.train.seed)
    d, t = exp.data, exp.train
    model = SparseRCNN(exp.model).to(DEVICE)
    initial_proposals = model.proposal_boxes.detach().cpu().clone()
    criterion = SetCriterion(exp.model.num_classes, exp.loss)
    opt = torch.optim.Adam(model.parameters(), lr=t.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=t.lr_step, gamma=t.lr_gamma)
    train_loader = make_loader(splits["train"], d, shuffle=True, seed=t.seed)
    val_loader = make_loader(splits["val"], d, shuffle=False)

    history = {"train_loss": [], "train_stage_losses": [], "val_map": [], "lr": [], "epoch_seconds": []}
    best = {"map": -1.0, "epoch": 0, "state": None}
    t_start = time.time()
    for epoch in range(1, t.max_epochs + 1):
        t0 = time.time()
        model.train()
        sums, steps = {}, 0
        for images, targets in train_loader:
            images = images.to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in tg.items()} for tg in targets]
            loss, logs = criterion(model(images), targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            opt.step()
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + v
            steps += 1
            if steps == t.max_steps_per_epoch:
                break
        history["lr"].append(opt.param_groups[0]["lr"])
        sched.step()
        history["train_loss"].append(sums["total"] / steps)
        history["train_stage_losses"].append({k: v / steps for k, v in sums.items() if k != "total"})

        val_map = evaluate(model, val_loader, d.img_size, CLASS_NAMES, full=False)["map"]
        history["val_map"].append(val_map)
        history["epoch_seconds"].append(time.time() - t0)
        improved = val_map > best["map"]
        if improved:
            best = {"map": val_map, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        if verbose:
            print(f"  epoch {epoch:2d} | train loss {history['train_loss'][-1]:.4f} | val mAP {val_map:.4f}"
                  f"{' *' if improved else ''} | lr {history['lr'][-1]:.1e} | {time.time() - t0:.1f}s", flush=True)
        if epoch - best["epoch"] >= t.patience:
            print(f"  early stop: no val mAP improvement for {t.patience} epochs", flush=True)
            break

    test_loader = make_loader(splits["test"], d, shuffle=False)
    last_test = evaluate(model, test_loader, d.img_size, CLASS_NAMES)["map"]
    model.load_state_dict(best["state"])
    test = evaluate(model, test_loader, d.img_size, CLASS_NAMES)
    test["map_last_epoch"] = last_test
    history.update(best_epoch=best["epoch"], best_val_map=best["map"], epochs_run=epoch,
                   train_minutes=(time.time() - t_start) / 60)
    return {"test": test, "history": history, "model": model, "state": best["state"],
            "initial_proposals": initial_proposals,
            "trained_proposals": model.proposal_boxes.detach().cpu().clone()}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _draw_boxes(ax, boxes_xyxy, labels, scores=None, dashed=False):
    for i, (box, lab) in enumerate(zip(boxes_xyxy, labels)):
        x1, y1, x2, y2 = [float(v) for v in box]
        color = COLORS[int(lab)]
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.6, ec=color,
                                        ls="--" if dashed else "-"))
        text = CLASS_NAMES[int(lab)] + (f" {float(scores[i]):.2f}" if scores is not None else "")
        ax.text(x1, y1 - 2, text, color="white", fontsize=6.5, va="bottom",
                bbox=dict(facecolor=color, alpha=0.8, pad=0.8, lw=0))


def plot_dataset_samples(split, path, n=5):
    x, y = split
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.8))
    for ax, img, tg in zip(axes, x[:n], y[:n]):
        ax.imshow(img)
        _draw_boxes(ax, tg["boxes"], tg["labels"])
        ax.axis("off")
    fig.suptitle("Synthetic training images with ground-truth boxes")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def plot_predictions(model, split, path, img_size, n=4):
    """Top row: ground truth. Bottom row: final-stage detections with score > SCORE_THRESHOLD."""
    model.eval()
    x, y = split
    images = torch.from_numpy(x[:n]).permute(0, 3, 1, 2).float().div(255).to(DEVICE)
    logits, boxes = model(images)
    preds = postprocess(logits[-1], boxes[-1], img_size, SCORE_THRESHOLD)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.4))
    for i in range(n):
        for row in (0, 1):
            axes[row, i].imshow(x[i])
            axes[row, i].axis("off")
        _draw_boxes(axes[0, i], y[i]["boxes"], y[i]["labels"])
        _draw_boxes(axes[1, i], preds[i]["boxes"], preds[i]["labels"], preds[i]["scores"])
    axes[0, 0].set_title("ground truth", fontsize=9, loc="left")
    axes[1, 0].set_title(f"predictions (score > {SCORE_THRESHOLD})", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_proposals(image, initial, trained, path):
    """The 20 learnable proposal boxes before and after training, on one test image."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.7))
    size = image.shape[0]
    for ax, boxes, title in zip(axes, (initial, trained), ("before training", "after training")):
        ax.imshow(image)
        for box in to_corners(boxes) * size:
            x1, y1, x2, y2 = box.tolist()
            ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1, ec="#f28e2b", alpha=0.8))
        ax.set_title(f"learnable proposals, {title}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_curves(runs_by_label, path):
    """runs_by_label: {label: [history, ...]} -> train loss and validation mAP per epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, (label, hists) in enumerate(runs_by_label.items()):
        for j, h in enumerate(hists):
            ep = range(1, len(h["train_loss"]) + 1)
            kw = dict(color=f"C{i}", alpha=0.8, lw=1.2, label=label if j == 0 else None)
            axes[0].plot(ep, h["train_loss"], **kw)
            axes[1].plot(ep, h["val_map"], **kw)
    axes[0].set_title("training loss (all stages)", fontsize=10)
    axes[0].set_yscale("log")
    axes[1].set_title("validation mAP@[.5:.95], final stage", fontsize=10)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("One line per seed")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_stage_map(stage_maps, path):
    """stage_maps: {label: (mean per stage, std per stage)}."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    width = 0.8 / len(stage_maps)
    for i, (label, (m, s)) in enumerate(stage_maps.items()):
        xs = np.arange(len(m)) + (i - (len(stage_maps) - 1) / 2) * width
        ax.bar(xs, m, width, yerr=s, capsize=3, label=label)
    ax.set_xticks(range(len(m)))
    ax.set_xticklabels([f"stage {k + 1}" for k in range(len(m))])
    ax.set_ylabel("test mAP@[.5:.95]")
    ax.set_title("Test mAP after each refinement stage (mean ± std over seeds)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
