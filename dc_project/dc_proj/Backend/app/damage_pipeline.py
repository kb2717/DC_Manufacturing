import torch

from app.config import load_config
from app.models.Resnet50.resnet50 import classify
from app.models.Resnet50.resnet50 import load_model as load_resnet
from app.models.YOLO26m.yolo26m import detect, predict, segmented_img
from app.models.YOLO26m.yolo26m import load_model as load_yolo
from app.utils.report import describe, load_vlm

_models = {}


def load_models(config=None):
    global _models
    if _models:
        return _models

    config = config or load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _models = {
        "resnet": load_resnet(config, device),
        "yolo": load_yolo(config),
        "vlm": load_vlm(config),
        "device": device,
        "conf": float((config.get("inference") or {}).get("conf", 0.25)),
    }
    return _models


def analyze(image_path):
    """Run the full pipeline on a single image.

    Returns:
        (segmented_image_or_None, classification_str, damage_text, description_text)
    """
    if image_path is None:
        return None, "Please upload an image.", "", ""

    m = load_models()
    label, confidence = classify(image_path, m["resnet"], m["device"])
    classification = f"{label} ({confidence:.1f}%)"

    # Run YOLO once and reuse the result for both the damage list and the overlay.
    # The detector runs on every image (it is cheap) so that a classifier
    # mistake cannot hide clearly visible damage; only the costly VLM report is
    # skipped for undamaged cars.
    results = predict(image_path, m["yolo"], conf=m["conf"])
    damage_list = detect(image_path, m["yolo"], results=results)

    if label == "undamaged":
        if not damage_list:
            return None, classification, "No damage detected.", "No damage detected."
        classification = (f"damaged — the detector found damage "
                          f"(classifier said undamaged, {confidence:.1f}%)")

    seg_img = segmented_img(image_path, m["yolo"], results=results)

    if not damage_list:
        msg = ("The classifier flagged this vehicle as damaged, but no damage region "
               "passed the detection threshold. Manual review recommended.")
        return seg_img, classification, "No damage region localized.", msg

    damage_text = "\n".join(f"- {d['class']} (conf: {d['confidence']})" for d in damage_list)
    description = describe(seg_img, damage_list, m["vlm"])

    return seg_img, classification, damage_text, description


if __name__ == "__main__":
    from app.api.router import create_interface

    demo = create_interface()
    demo.launch(share=True)