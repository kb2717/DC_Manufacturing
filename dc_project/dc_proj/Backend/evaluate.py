"""Score the trained models on VehiDE's validation split.

Run from Backend/:
    python evaluate.py                 # both models
    python evaluate.py --skip-yolo     # classifier only
    python evaluate.py --imgsz 1280    # detector at the resolution it was trained at

Writes results/metrics.json and results/metrics.md (a table ready for the README).
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.config import load_config, resolve_path
from app.models.Resnet50.resnet50 import (
    TransformSubset,
    build_datasets,
    load_model as load_resnet,
    test_transforms,
)


def evaluate_resnet(config, batch_size=64, workers=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_resnet(config, device)
    _, test_ds = build_datasets(config)
    loader = DataLoader(TransformSubset(test_ds, test_transforms),
                        batch_size=batch_size, shuffle=False, num_workers=workers)

    tp = tn = fp = fn = 0
    with torch.inference_mode():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(1).cpu()
            tp += int(((preds == 1) & (labels == 1)).sum())
            tn += int(((preds == 0) & (labels == 0)).sum())
            fp += int(((preds == 1) & (labels == 0)).sum())
            fn += int(((preds == 0) & (labels == 1)).sum())

    total = max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "test_images": total,
        "accuracy": (tp + tn) / total,
        "precision_damaged": precision,
        "recall_damaged": recall,
        "f1_damaged": 2 * precision * recall / max(precision + recall, 1e-9),
        "specificity_undamaged": tn / max(tn + fp, 1),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def evaluate_yolo(config, imgsz=1280, batch=8):
    from app.models.YOLO26m.yolo26m import load_model as load_yolo

    model = load_yolo(config)
    data = str(resolve_path(config["paths"]["data"]))
    device = 0 if torch.cuda.is_available() else "cpu"
    metrics = model.val(data=data, split="val", imgsz=imgsz, batch=batch,
                        device=device, plots=True, project="results", name="yolo_val",
                        exist_ok=True)

    names = model.names
    seg = metrics.seg
    per_class = {}
    for i, c in enumerate(metrics.ap_class_index):
        p, r, ap50, ap = seg.class_result(i)
        per_class[names[int(c)]] = {"precision": p, "recall": r, "mAP50": ap50, "mAP50-95": ap}
    return {
        "mask": {"precision": seg.mp, "recall": seg.mr, "mAP50": seg.map50, "mAP50-95": seg.map},
        "box": {"precision": metrics.box.mp, "recall": metrics.box.mr,
                "mAP50": metrics.box.map50, "mAP50-95": metrics.box.map},
        "per_class_mask": per_class,
    }


def to_markdown(out):
    lines = []
    if "resnet" in out:
        r = out["resnet"]
        lines += ["### ResNet-50 damaged / undamaged classifier", "",
                  "| Metric | Value |", "|---|---|",
                  f"| Test images | {r['test_images']} |",
                  f"| Accuracy | {r['accuracy']:.3f} |",
                  f"| Precision (damaged) | {r['precision_damaged']:.3f} |",
                  f"| Recall (damaged) | {r['recall_damaged']:.3f} |",
                  f"| F1 (damaged) | {r['f1_damaged']:.3f} |",
                  f"| Specificity (undamaged) | {r['specificity_undamaged']:.3f} |", ""]
    if "yolo" in out:
        y = out["yolo"]
        lines += ["### YOLO26 damage segmentation (VehiDE val, mask metrics)", "",
                  "| Class | Precision | Recall | mAP50 | mAP50-95 |", "|---|---|---|---|---|"]
        m = y["mask"]
        lines.append(f"| All | {m['precision']:.3f} | {m['recall']:.3f} | {m['mAP50']:.3f} | {m['mAP50-95']:.3f} |")
        for name, v in y["per_class_mask"].items():
            lines.append(f"| {name.capitalize()} | {v['precision']:.3f} | {v['recall']:.3f} | "
                         f"{v['mAP50']:.3f} | {v['mAP50-95']:.3f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-resnet", action="store_true")
    ap.add_argument("--skip-yolo", action="store_true")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    config = load_config()
    out = {}
    if not args.skip_resnet:
        out["resnet"] = evaluate_resnet(config, workers=args.workers)
    if not args.skip_yolo:
        out["yolo"] = evaluate_yolo(config, imgsz=args.imgsz)

    Path("results").mkdir(exist_ok=True)
    Path("results/metrics.json").write_text(json.dumps(out, indent=2))
    md = to_markdown(out)
    Path("results/metrics.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
