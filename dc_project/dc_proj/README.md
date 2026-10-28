# Car Damage Detection

This project is a deep learning pipeline for automated vehicle damage assessment. A single uploaded car photo is classified as damaged or undamaged, damage is localized and categorized, and a natural-language damage report is generated — end to end, with no manual assessment required.

## Overview

The pipeline chains three model stages rather than one end-to-end model, so each stage can be trained, evaluated, and improved independently:

1. **Classify** whether the vehicle is damaged at all (ResNet-50)
2. **Detect and segment** the specific damage regions, if any (YOLO26-seg)
3. **Report** location, severity, and repair complexity in natural language (Qwen3.5-VL)

## Key Features

- 🔍 Binary damage classification using a fine-tuned ResNet-50
- 🎯 Multi-class damage detection and instance segmentation across 7 damage categories using YOLOv26m
- 🧠 Natural-language damage reporting using Qwen3.5-VL
- 🖥️ Gradio interface for uploading an image and viewing results end to end
- ⚙️ Single config-driven setup for both training and inference paths
- 🧩 Early-exit design — undamaged images skip detection and reporting entirely

## Architecture

```
                  Uploaded car image
                          │
                          ▼
                ┌───────────────────┐
                │  ResNet-50        │
                │  binary classifier│
                └─────────┬─────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        "undamaged"               "damaged"
              │                       │
              ▼                       ▼
     Return "no damage"      ┌───────────────────┐
                              │  YOLOv26m detector │
                              │  detect + segment  │
                              └─────────┬───────────┘
                                        ▼
                              ┌───────────────────┐
                              │  Qwen3.5-VL         │
                              │  report generator   │
                              └─────────┬───────────┘
                                        ▼
                                 Gradio interface
                     (segmented image, class, damage list, report)
```

## Repository Structure

```
.
├── Backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── router.py              # Gradio interface definition
│   │   ├── models/
│   │   │   ├── Resnet50/
│   │   │   │   └── resnet50.py        # classifier: model, training, inference
│   │   │   └── YOLO26m/
│   │   │       ├── data_converter.py  # VIA JSON -> YOLO segmentation dataset
│   │   │       └── yolo26m.py         # detector: model, training, inference
│   │   ├── utils/
│   │   │   └── report.py              # Qwen3.5-VL report generation
│   │   ├── damage_pipeline.py         # loads models, chains the pipeline
│   │   └── config.py                  # shared config loader
│   ├── main.py                        # entry point — launches the Gradio app
│   ├── evaluate.py                    # scores both models on VehiDE val
│   ├── download_weights.py            # fetches trained weights into weights/
│   ├── config.yaml.example
│   └── requirements.txt
│
├── notebooks/
│   └── train_vehide.ipynb             # GPU retraining on VehiDE + undamaged cars
│
├── assets/                            # curves and confusion matrix
│
├── .gitignore
└── README.md
```

## Core Modules

### 🔍 Classifier — `Backend/app/models/Resnet50/resnet50.py`

ResNet-50 with a custom head (`Dropout → Linear(2)`) fine-tuned to answer one binary question: is this vehicle damaged? If undamaged, the pipeline exits immediately and skips detection and reporting.

### 🎯 Detector — `Backend/app/models/YOLO26m/yolo26m.py`

Localizes and segments damage across 7 classes: broken glass, broken lamp, dent, hole, lost part, paint scratch, and torn. Returns bounding boxes, class labels, confidence scores, and an annotated segmentation.

### 🧠 Reporter — `Backend/app/utils/report.py`

Passes the segmented image and detected damage classes to Qwen3.5-VL, prompted as an expert insurance damage assessor, returning damage location and type, severity, estimated repair complexity, and a summary.

### 🧩 Pipeline — `Backend/app/damage_pipeline.py`

Loads all three models once and chains classification → detection → reporting, with the early-exit branch for undamaged vehicles.

### 🖥️ Interface — `Backend/app/api/router.py` + `Backend/main.py`

`router.py` builds the Gradio `Interface` around `damage_pipeline.analyze()`; `main.py` launches it.

## Technology Stack

| Category | Technology |
|---|---|
| Language | Python |
| Classification | ResNet-50 (torchvision) |
| Detection & Segmentation | YOLOv26m (Ultralytics) |
| Vision-Language Reporting | Qwen3.5-VL (via Hugging Face Router) |
| Interface | Gradio |
| Config Management | YAML |
| HTTP Client | httpx |

## Performance & Results

### YOLO26 detection and segmentation (VehiDE validation, mask metrics)

The released weights are YOLO26**s**-seg trained at 1280px for 150 epochs. The folder keeps its historical `YOLO26m` name; set `models.yolo26m_pt` to `yolo26m-seg.pt` to train the medium model instead.

| Class | Mask precision | Mask recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| All | 0.665 | 0.501 | 0.523 | 0.308 |
| Broken glass | 0.856 | 0.789 | 0.836 | 0.608 |
| Broken lamp | 0.669 | 0.564 | 0.561 | 0.290 |
| Dent | 0.618 | 0.339 | 0.361 | 0.159 |
| Hole | 0.663 | 0.481 | 0.518 | 0.284 |
| Lost part | 0.787 | 0.677 | 0.719 | 0.511 |
| Paint scratch | 0.490 | 0.289 | 0.282 | 0.119 |
| Torn | 0.574 | 0.368 | 0.384 | 0.186 |

Overall mask precision across all classes reaches 0.99 at a confidence threshold of 1.0, and overall mask recall reaches 0.70 at a confidence threshold of 0.0.


### Classifier note

The released ResNet-50 was trained on VehiDE alone. VehiDE contains only damaged cars, so that model labels almost any photo "damaged" (in testing it did so for a portrait, a cheetah and an undamaged bus), and the early-exit branch rarely fires. Retrain it with an undamaged-car dataset using `notebooks/train_vehide.ipynb`; `evaluate.py` reports its specificity on undamaged cars so the improvement is measurable.

### Pipeline in action

**Undamaged vehicle** — classifier correctly identifies no damage and the pipeline exits before detection:
<img width="1600" height="611" alt="undamaged" src="https://github.com/user-attachments/assets/1901dd81-e530-43b9-bccb-588917d0c7da" />


[Undamaged classification result]

**Damaged vehicle** — front bumper lost parts and torn material detected, segmented, and described:
<img width="1600" height="779" alt="image" src="https://github.com/user-attachments/assets/c02c233a-2b1b-4ea6-bd3c-c905541953d5" />


[Damaged classification and report]

## Key Highlights & Design Decisions

- **Three independent stages instead of one end-to-end model.** Classification, detection, and reporting are separate, swappable components. Each can be retrained or upgraded (e.g. a newer YOLO version, or a different VLM) without touching the others.
- **Early-exit for efficiency.** Most uploaded images are undamaged. Running detection and a VLM call on every image would waste compute and API cost, so the classifier gates the rest of the pipeline — undamaged images return instantly, damaged ones proceed to the heavier stages.
- **VLM-generated reports instead of rule-based templates.** Rather than hand-writing severity/cost logic per damage class, Qwen3.5-VL reasons directly over the segmented image and detected classes, producing a contextual, insurance-style writeup that generalizes to damage combinations no template could anticipate.
- **Single config, multiple concerns.** One `config.yaml` covers both inference paths and training/data-prep paths, so switching between running the app and retraining a model doesn't mean juggling separate config files.
- **Refactored from a single script into a modular package.** The original pipeline was one file; it's now split by responsibility (`models/`, `utils/`, `api/`) so each model's training and inference code lives with the model it belongs to, and the orchestration layer stays decoupled from both.

## Getting Started

### 1. Install

```bash
cd Backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config.yaml.example config.yaml
cp .env.example .env        # optional: add HF_TOKEN to enable the written report
```

Without `HF_TOKEN` the app still classifies, detects and segments damage; only the Qwen report is replaced by a short notice.

### 3. Get the weights

```bash
python download_weights.py
```

This puts `resnet50_damage.pth` and `yolo26_damage.pt` in `Backend/weights/`. It downloads them from the GitHub Release `v1.0` if one exists, otherwise it extracts the originally trained weights from this repository's git history (needs a full clone, not `--depth 1`).

### 4. Run

```bash
python main.py              # http://127.0.0.1:7860
APP_SHARE=1 python main.py  # also prints a public *.gradio.live link
```

## Retraining on VehiDE (+ an undamaged-car dataset)

Training needs a GPU. Open [`notebooks/train_vehide.ipynb`](notebooks/train_vehide.ipynb) on Kaggle (GPU T4/P100) or Colab and attach:

| Dataset | Used for |
|---|---|
| [VehiDE](https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection) | YOLO26 segmentation (train + official val split) and the *damaged* class of the classifier |
| An undamaged-car set, e.g. [Car Damage Dataset](https://www.kaggle.com/datasets/vinayjose/car-damage-dataset) (`01-whole` folders) or Stanford Cars | The *undamaged* class of the classifier — VehiDE contains only damaged cars |

The notebook finds both datasets, writes `config.yaml`, then runs the same scripts you can run locally from `Backend/`:

```bash
python -m app.models.YOLO26m.data_converter   # VehiDE VIA JSON -> YOLO polygons (official val split if configured)
python -m app.models.Resnet50.resnet50        # damaged / undamaged classifier
python -m app.models.YOLO26m.yolo26m          # YOLO26 segmentation (resumes from last.pt if interrupted)
python evaluate.py --imgsz 1280               # scores both on VehiDE val -> results/metrics.md
```

`PROFILE = "fast"` (640px, 60 epochs) fits one 12-hour Kaggle session; `"full"` (1280px, 150 epochs) reproduces the released model.
## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Qwen](https://github.com/QwenLM/Qwen) via Hugging Face Router
- [Gradio](https://www.gradio.app/)
