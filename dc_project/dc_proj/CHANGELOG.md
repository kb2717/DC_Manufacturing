# Changelog

## Integration fixes

**App would not start**
- `config.yaml.example` pointed at weight files that never existed. It now points to `weights/resnet50_damage.pth` and `weights/yolo26_damage.pt`, and `Backend/download_weights.py` puts the trained weights there (GitHub Release, or the repo's own git history as a fallback).

**Pipeline (`damage_pipeline.py`, `yolo26m.py`)**
- YOLO ran twice per image (once for the list, once for the overlay); it now runs once and both reuse the result.
- Detected classes are shown in English (VehiDE's raw labels are Vietnamese).
- When the classifier says "damaged" but YOLO finds nothing, the app says so instead of sending an empty list to the VLM.
- Confidence is shown as `damaged (97.3%)` instead of a raw float; an empty submit no longer crashes.
- `inference.conf` in the config sets the detection threshold.

**UI (`router.py`)**
- The Submit button was never disabled before an upload: the lookup expected exactly one button, but `gr.Interface` also creates Clear.

**Training**
- `yolo26m.py`: `auto_augment=False` crashed training on current Ultralytics; removed.
- `yolo26m.py`: re-running after a *finished* run silently trained on Ultralytics' default coco8 dataset. It now resumes only an interrupted run and otherwise trains on VehiDE.
- `yolo26m.py`: GPU `device=0` was hard-coded; the referenced `hyp.yaml` did not exist. Device, epochs, image size and batch now come from `config.yaml`; the hyp file is optional. Best weights are copied into `weights/` automatically.
- `data_converter.py`: uses VehiDE's official validation split when configured; finds images in nested folders; drops degenerate polygons (< 3 points); writes English class names in the same index order as the released model.
- `resnet50.py`: adds undamaged-car images from a second dataset (VehiDE has none), splits them reproducibly, weights the loss for class imbalance, and reads batch/epochs/workers from config. The model and transforms are unchanged.

## Accuracy fixes (found from a misclassified crash photo)

- **The classifier judges framing, not damage.** The released ResNet-50 calls close-ups "damaged" and whole-car shots "undamaged" (a front-crashed pickup scored undamaged 99.8%; a crop of its bumper scored damaged 99.8%). VehiDE is mostly close-ups, so framing became the shortcut.
  - `damage_pipeline.py`: the detector now runs on every image; a car is reported damaged if either model finds damage. Only the VLM call is skipped for undamaged cars.
  - `resnet50.py`: new `damaged_dirs` (whole-car photos of damaged cars, e.g. `00-damage`) and random zoom-crops during training, so both classes appear at every framing.
- **Small photos were upscaled to 1280px**, which hid damage from the detector. `yolo26m.py` now infers at the photo's own size (640–1280px). The crash photo went from no detections to "lost parts 0.62".

## New
- `Backend/evaluate.py` — scores both models on VehiDE val and writes `results/metrics.md`.
- `notebooks/train_vehide.ipynb` — GPU retraining on Kaggle/Colab with VehiDE + an undamaged-car dataset.
