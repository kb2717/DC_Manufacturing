import json
import os
import random
import shutil

from PIL import Image, ImageFile

from app.config import load_config, resolve_path

ImageFile.LOAD_TRUNCATED_IMAGES = True

SPLIT_RATIO = 0.8
RANDOM_SEED = 42


def main():
    config = load_config()

    paths = config["paths"]
    raw_img_dir = resolve_path(paths["img_dir"])
    json_path = resolve_path(paths["json_path"])
    out_dir = resolve_path(paths["yolo_dataset"])

    for split in ["train", "val"]:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    with open(json_path, "r") as f:
        jsondata = json.load(f)

    damaged_data, classes = _collect_damaged_regions(jsondata)

    # VehiDE ships an official validation split — use it when configured, so
    # reported metrics are comparable with other VehiDE results.
    val_json = paths.get("val_json_path")
    val_img_dir = paths.get("val_img_dir")
    if val_json and val_img_dir and resolve_path(val_json).exists():
        with open(resolve_path(val_json), "r") as f:
            val_data, val_classes = _collect_damaged_regions(json.load(f))
        classes |= val_classes
        train_data = damaged_data
        val_index = _index_images(resolve_path(val_img_dir))
        print(f"Using official VehiDE split: {len(train_data)} train / {len(val_data)} val")
    else:
        random.seed(RANDOM_SEED)
        random.shuffle(damaged_data)
        split_idx = int(SPLIT_RATIO * len(damaged_data))
        train_data = damaged_data[:split_idx]
        val_data = damaged_data[split_idx:]
        val_index = None
        print(f"Random {SPLIT_RATIO:.0%} split: {len(train_data)} train / {len(val_data)} val")

    # Order classes by their English name — the same index order as the
    # released weights (0 broken glass ... 6 torn).
    from app.models.YOLO26m.yolo26m import VEHIDE_CLASSES

    ordered = sorted(classes, key=lambda c: VEHIDE_CLASSES.get(c, c))
    class_map = {cls_name: idx for idx, cls_name in enumerate(ordered)}

    train_index = _index_images(raw_img_dir)
    n_tr = _process_data(train_data, "train", train_index, out_dir, class_map)
    n_va = _process_data(val_data, "val", val_index or train_index, out_dir, class_map)
    print(f"Wrote {n_tr} train and {n_va} val images to {out_dir}")

    _write_data_yaml(out_dir, class_map)


def _index_images(root):
    """Map file name -> full path for every image under root (searched recursively)."""
    index = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                index.setdefault(name, os.path.join(dirpath, name))
    return index


def _collect_damaged_regions(jsondata):
    damaged_data = []
    classes = set()

    if isinstance(jsondata, dict):
        for img_id, img_data in jsondata.items():
            if not isinstance(img_data, dict):
                continue
            regions = img_data.get("regions", [])
            if len(regions) == 0:
                continue
            filename = img_data.get("filename", img_data.get("name", img_id))
            damaged_data.append((filename, regions))
            for region in regions:
                cls_name = _region_class(region)
                if cls_name:
                    classes.add(cls_name)

    return damaged_data, classes


def _region_class(region):
    """VehiDE stores the label as region['class']; standard VIA uses region_attributes."""
    cls_name = region.get("class")
    if not cls_name:
        attrs = region.get("region_attributes") or {}
        cls_name = next((v for v in attrs.values() if isinstance(v, str) and v), None)
    return cls_name


def _process_data(data, split_name, image_index, out_dir, class_map):
    written = 0
    for filename, regions in data:
        img_path = image_index.get(os.path.basename(filename))
        if not img_path:
            continue

        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception:
            continue

        filename = os.path.basename(filename)
        shutil.copy(img_path, os.path.join(out_dir, "images", split_name, filename))

        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(out_dir, "labels", split_name, label_filename)

        with open(label_path, "w") as lf:
            for region in regions:
                line = _region_to_yolo_line(region, class_map, width, height)
                if line:
                    lf.write(line + "\n")
        written += 1
    return written


def _region_to_yolo_line(region, class_map, width, height):
    cls_name = _region_class(region)
    if not cls_name or cls_name not in class_map:
        return None

    cls_idx = class_map[cls_name]
    line_data = [str(cls_idx)]

    shape = region.get("shape_attributes") or region
    if "all_x" in shape and "all_y" in shape:
        xs, ys = shape["all_x"], shape["all_y"]
    elif "all_points_x" in shape and "all_points_y" in shape:
        xs, ys = shape["all_points_x"], shape["all_points_y"]
    else:
        xs, ys = [], []

    for x, y in zip(xs, ys):
        norm_x = min(max(x / width, 0.0), 1.0)
        norm_y = min(max(y / height, 0.0), 1.0)
        line_data.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])

    # A segmentation polygon needs at least 3 points (class id + 6 coordinates).
    if len(line_data) < 7:
        return None
    return " ".join(line_data)


def _write_data_yaml(out_dir, class_map):
    from app.models.YOLO26m.yolo26m import VEHIDE_CLASSES

    # Train with English class names so the model's own labels are readable.
    names_str = "\n".join(
        f"  {idx}: {VEHIDE_CLASSES.get(cls_name, cls_name)}" for cls_name, idx in class_map.items()
    )
    yaml_content = f"""path: {out_dir}
train: images/train
val: images/val

names:
{names_str}
"""
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write(yaml_content)


if __name__ == "__main__":
    main()