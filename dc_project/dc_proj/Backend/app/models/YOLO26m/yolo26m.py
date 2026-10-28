import os

from PIL import Image
from ultralytics import YOLO

from app.config import load_config, resolve_path

RUN_NAME = "26m_pro"

# VehiDE labels its classes in Vietnamese. Models trained straight from the raw
# annotations carry these names, so they are translated for display.
VEHIDE_CLASSES = {
    "vo_kinh": "broken glass",
    "be_den": "broken lamp",
    "mop_lom": "dent",
    "thung": "hole",
    "mat_bo_phan": "lost parts",
    "tray_son": "paint scratch",
    "rach": "torn",
}


def load_model(config):
    yolo_path = resolve_path(config["paths"]["yolo"])
    if not yolo_path.exists():
        raise FileNotFoundError(f"Weights not found at: {yolo_path}")
    model = YOLO(str(yolo_path))
    # Show English names in the overlay and the damage list.
    names = model.model.names
    for k, v in list(names.items()):
        names[k] = VEHIDE_CLASSES.get(v, v)
    return model


def _infer_size(image_path, max_size=1280, min_size=640):
    """Inference size that does not upscale small photos.

    The model was trained at 1280px, but blowing a 500px photo up to 1280 makes
    damage look unlike anything it was trained on and detections vanish.
    """
    try:
        with Image.open(image_path) as img:
            side = max(img.size)
    except Exception:
        return max_size
    side = min(max(side, min_size), max_size)
    return int(-(-side // 32) * 32)  # round up to a multiple of 32


def predict(image_path, yolo_model, conf=0.25):
    """Run inference once; pass the result to detect() and segmented_img()."""
    return yolo_model(image_path, conf=conf, imgsz=_infer_size(image_path), verbose=False)


def detect(image_path, yolo_model, results=None):
    """Run detection and return a list of {'class': str, 'confidence': float} dicts."""
    if results is None:
        results = predict(image_path, yolo_model)
    damage_found = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            cls_name = yolo_model.names[cls_id]
            confidence = round(float(box.conf.item()), 2)
            damage_found.append({"class": cls_name, "confidence": confidence})
    return damage_found


def segmented_img(image_path, yolo_model, results=None):
    """Run inference and return the annotated/segmented image as a PIL Image."""
    if results is None:
        results = predict(image_path, yolo_model)
    annotated = results[0].plot()
    return Image.fromarray(annotated[..., ::-1])


def _train_device(config):
    import torch

    device = str((config.get("training") or {}).get("device", "auto"))
    if device == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    return device


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    config = load_config()
    project_dir = resolve_path(config["paths"]["runs_segment"])
    last_weights = os.path.join(str(project_dir), RUN_NAME, "weights", "last.pt")

    if os.path.exists(last_weights) and _is_unfinished(last_weights):
        model = YOLO(last_weights)
        model.train(resume=True)
        _export_best(model, config)
        return

    train_cfg = config.get("training") or {}
    hyp = resolve_path(config["paths"].get("hyperparameters", "app/models/YOLO26m/hyp.yaml"))
    extra = {"cfg": str(hyp)} if hyp.exists() else {}

    model = YOLO(config["models"]["yolo26m_pt"])
    model.train(
        data=str(resolve_path(config["paths"]["data"])),
        epochs=int(train_cfg.get("yolo_epochs", 100)),
        imgsz=int(train_cfg.get("yolo_imgsz", 640)),
        batch=int(train_cfg.get("yolo_batch", 16)),
        workers=int(train_cfg.get("workers", 8)),
        device=_train_device(config),
        project=str(project_dir),
        name=RUN_NAME,
        patience=30,
        optimizer="MuSGD",
        amp=True,
        cos_lr=True,
        retina_masks=True,
        overlap_mask=False,
        erasing=0.0,
        exist_ok=True,
        **extra,
    )
    _export_best(model, config)


def _is_unfinished(ckpt_path):
    """True if last.pt still holds optimizer state, i.e. training was interrupted.

    A finished run strips that state; resuming it would make Ultralytics start a
    fresh run on its default dataset instead of VehiDE.
    """
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("epoch", -1) >= 0 and ckpt.get("optimizer") is not None


def _export_best(model, config):
    """Copy the run's best.pt to paths.yolo so the app picks it up."""
    import shutil

    best = os.path.join(str(model.trainer.save_dir), "weights", "best.pt")
    dest = resolve_path(config["paths"]["yolo"])
    if os.path.exists(best):
        os.makedirs(dest.parent, exist_ok=True)
        shutil.copy(best, dest)
        print(f"Best weights copied to {dest}")


if __name__ == "__main__":
    main()