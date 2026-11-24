"""Train every experiment over several seeds, evaluate on the held-out test set, write results/.

    python src/run_experiments.py                        # default experiments x 3 seeds (GPU recommended)
    python src/run_experiments.py --only more_data       # one experiment; merges into metrics.json
    python src/run_experiments.py --smoke                # 1 seed, 2 tiny epochs -> results/smoke/
"""
import argparse
import json
import platform
import sys
from dataclasses import replace

import numpy as np
import torch

from config import (CLASS_NAMES, DEFAULT_EXPERIMENTS, DEVICE, EXPERIMENTS, RESULTS_DIR, with_seed)
from data import make_splits, split_stats
from model import SparseRCNN, count_params
from train import fit, plot_curves, plot_dataset_samples, plot_predictions, plot_proposals, plot_stage_map

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def mean_std(values):
    return float(np.mean(values)), float(np.std(values))


def fmt(values, digits=3):
    m, s = mean_std(values)
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def summarize(runs):
    keys = ["map", "map_50", "map_75", "mar_100", "map_thr0.3", "count_exact", "count_over",
            "count_under", "map_last_epoch"]
    out = {k: mean_std([r["test"][k] for r in runs]) for k in keys}
    out["ap_per_class"] = {c: mean_std([r["test"]["ap_per_class"][c] for r in runs]) for c in CLASS_NAMES}
    stages = np.array([r["test"]["map_per_stage"] for r in runs])
    out["map_per_stage"] = (stages.mean(0).tolist(), stages.std(0).tolist())
    return out


def write_table(metrics, out_dir):
    head = ("| Model | Params | Test mAP@[.5:.95] ↑ | mAP@.5 ↑ | mAP@.75 ↑ | AR@100 ↑ | "
            "AP circle / square / triangle | Exact object count ↑ | Selected epoch |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    rows = []
    for name, m in metrics["models"].items():
        runs = m["runs"]
        ap = " / ".join(f"{np.mean([r['test']['ap_per_class'][c] for r in runs]):.3f}" for c in CLASS_NAMES)
        rows.append(f"| {m['label']} | {m['params'] / 1e6:.2f}M | {fmt([r['test']['map'] for r in runs])} | "
                    f"{fmt([r['test']['map_50'] for r in runs])} | {fmt([r['test']['map_75'] for r in runs])} | "
                    f"{fmt([r['test']['mar_100'] for r in runs])} | {ap} | "
                    f"{fmt([r['test']['count_exact'] for r in runs], 2)} | "
                    f"{', '.join(str(r['history']['best_epoch']) for r in runs)} |")
    ev = metrics["evaluation"]
    note = (f"\nMean ± std over seeds {ev['seeds']}. Test set: {ev['test_images']:,} freshly generated images "
            f"({ev['test_objects']:,} objects), never used for checkpoint selection (that uses the "
            f"{ev['val_images']}-image validation split). mAP is COCO-style over all 20 detections per image. "
            f"'Exact object count' is the share of test images with exactly as many detections above score "
            f"0.3 as objects. Hardware: {ev['environment']['device']}.\n")
    text = head + "\n".join(rows) + "\n" + note
    (out_dir / "results_table.md").write_text(text, encoding="utf-8")
    print("\n" + text)


def write_summary_figures(metrics, fig_dir):
    models = metrics["models"]
    plot_curves({m["label"]: [r["history"] for r in m["runs"]] for m in models.values()},
                fig_dir / "training_curves.png")
    plot_stage_map({m["label"]: tuple(m["summary"]["map_per_stage"]) for m in models.values()},
                   fig_dir / "map_per_stage.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), default=list(DEFAULT_EXPERIMENTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="1 seed, 2 epochs x 5 steps, small test set")
    args = ap.parse_args()

    out_dir = RESULTS_DIR / "smoke" if args.smoke else RESULTS_DIR
    fig_dir, ckpt_dir = out_dir / "figures", out_dir / "checkpoints"
    for p in (fig_dir, ckpt_dir):
        p.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds[:1] if args.smoke else args.seeds
    device_name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else platform.processor() or "CPU"
    print(f"device: {device_name} | torch {torch.__version__}", flush=True)

    metrics_path = out_dir / "metrics.json"
    merged = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {"models": {}}
    split_cache = {}

    for name in args.only:
        base = EXPERIMENTS[name]
        if args.smoke:
            base = replace(base, data=replace(base.data, n_test=64),
                           train=replace(base.train, max_epochs=2, max_steps_per_epoch=5))
        if args.epochs:
            base = replace(base, train=replace(base.train, max_epochs=args.epochs))
        key = (base.data.n_pool, base.data.n_test)
        if key not in split_cache:
            split_cache[key] = make_splits(base.data)
        splits = split_cache[key]
        stats = split_stats(splits)
        if "evaluation" not in merged or name == "sparse_rcnn":
            merged["evaluation"] = {"seeds": seeds, "val_images": stats["val"]["images"],
                                    "test_images": stats["test"]["images"],
                                    "test_objects": stats["test"]["objects"], "splits": stats,
                                    "environment": {"device": device_name, "torch": torch.__version__}}
            plot_dataset_samples(splits["train"], fig_dir / "dataset_samples.png")
        print(f"\n[{name}] {base.label} | train {stats['train']['images']} imgs / "
              f"{stats['train']['objects']} objects | val {stats['val']['images']} | test {stats['test']['images']}",
              flush=True)

        runs = []
        for seed in seeds:
            exp = with_seed(base, seed)
            print(f"[{name}] seed {seed} | {count_params(SparseRCNN(exp.model)):,} params", flush=True)
            res = fit(exp, splits)
            torch.save(res["state"], ckpt_dir / f"{name}_seed{seed}.pt")
            t = res["test"]
            print(f"  -> best epoch {res['history']['best_epoch']}/{res['history']['epochs_run']} | "
                  f"test mAP {t['map']:.4f} | mAP50 {t['map_50']:.4f} | per stage "
                  f"{', '.join(f'{v:.3f}' for v in t['map_per_stage'])} | exact count {t['count_exact']:.2f} | "
                  f"{res['history']['train_minutes']:.1f} min", flush=True)
            if seed == seeds[0]:
                plot_predictions(res["model"], splits["test"], fig_dir / f"{name}_predictions.png",
                                 base.data.img_size)
                plot_proposals(splits["test"][0][0], res["initial_proposals"], res["trained_proposals"],
                               fig_dir / f"{name}_proposals.png")
            runs.append({"seed": seed, "test": t, "history": res["history"],
                         "initial_proposals": res["initial_proposals"].tolist(),
                         "trained_proposals": res["trained_proposals"].tolist()})

        merged["models"][name] = {"label": base.label, "params": count_params(SparseRCNN(base.model)),
                                  "config": base.to_dict(), "runs": runs, "summary": summarize(runs)}
        metrics_path.write_text(json.dumps(merged, indent=1), encoding="utf-8")
        write_table(merged, out_dir)
        write_summary_figures(merged, fig_dir)


if __name__ == "__main__":
    main()
