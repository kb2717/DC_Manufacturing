import hashlib
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

from app.config import load_config, resolve_path

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASSES = ["undamaged", "damaged"]

test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def build_model(device, weights=None):
    """Build the ResNet-50 architecture with the custom binary head."""
    model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 2),
    )
    return model.to(device)


def load_model(config, device):
    """Build the architecture and load trained weights, ready for serving."""
    resnet_path = resolve_path(config["paths"]["resnet"])
    if not resnet_path.exists():
        raise FileNotFoundError(f"Weights not found at: {resnet_path}")

    model = build_model(device, weights=None)
    model.load_state_dict(torch.load(resnet_path, map_location=device))
    model.to(device).eval()
    return model


def classify(image_path, model, device, transform=test_transforms):
    """Classify an image as 'damaged' or 'undamaged'.

    Returns:
        (label: str, confidence_pct: float)
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not open image at {image_path}: {e}") from e

    tensor = transform(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)
        confidence, pred = torch.max(prob, 1)

    label = CLASSES[pred.item()]
    return label, confidence.item() * 100


IMG_EXTS = (".png", ".jpg", ".jpeg")


def list_images(root):
    """All image paths under root, searched recursively, in a stable order."""
    found = []
    for dirpath, _, files in os.walk(root):
        found += [os.path.join(dirpath, f) for f in files if f.lower().endswith(IMG_EXTS)]
    return sorted(found)


def is_holdout(path, fraction=0.2):
    """Deterministic train/test assignment for extra (undamaged) images.

    Hashing the file name keeps the split identical between training and
    evaluate.py, so the test images are never seen during training.
    """
    h = int(hashlib.md5(os.path.basename(path).encode()).hexdigest(), 16)
    return (h % 1000) < fraction * 1000


class VehicleDataset(Dataset):
    """Images labelled from a VehiDE-style JSON: regions present -> damaged (1).

    Images in img_folder that are missing from the JSON count as undamaged (0),
    as before. extra_undamaged / extra_damaged add photos from other datasets
    as class 0 / class 1.
    """

    def __init__(self, json_file, img_folder, extra_undamaged=(), extra_damaged=()):
        self.data = []

        label_map = {}
        if json_file:
            with open(json_file, "r") as f:
                jsondata = json.load(f)
            if isinstance(jsondata, dict):
                for img_id, img_data in jsondata.items():
                    if isinstance(img_data, dict):
                        filename = img_data.get("filename", img_data.get("name", img_id))
                        regions = img_data.get("regions", [])
                        label_map[os.path.basename(filename)] = 1 if len(regions) > 0 else 0

        if img_folder:
            for path in list_images(img_folder):
                label = label_map.get(os.path.basename(path), 0)
                self.data.append((path, label))

        self.data += [(path, 0) for path in extra_undamaged]
        self.data += [(path, 1) for path in extra_damaged]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img_path, label = self.data[i]
        img = Image.open(img_path).convert("RGB")
        return img, torch.tensor(label, dtype=torch.long)

    def label_counts(self):
        n_damaged = sum(label for _, label in self.data)
        return len(self.data) - n_damaged, n_damaged


class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.subset[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)


def _run_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()

            out = model(imgs)
            loss = criterion(out, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, predict = torch.max(out, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()

    return running_loss / total, 100 * correct / total


def build_datasets(config):
    """(train_dataset, test_dataset) of (PIL image, label) pairs.

    Damaged images come from VehiDE; undamaged ones from paths.undamaged_dirs.
    If VehiDE's validation split is configured it becomes the damaged half of the
    test set, so the classifier is scored on the same images as the detector.
    Otherwise everything is split 80/20 at random (seed 42), as before.
    """
    paths = config["paths"]
    img_dir = resolve_path(paths["img_dir"])
    json_path = resolve_path(paths["json_path"])

    def collect(key):
        found = []
        for d in paths.get(key) or []:
            if resolve_path(d).exists():
                found += list_images(resolve_path(d))
        return found

    undamaged = collect("undamaged_dirs")
    # Whole-car photos of damaged cars. VehiDE is mostly damage close-ups, so
    # without these the model learns "close-up = damaged, whole car = undamaged".
    damaged = collect("damaged_dirs")
    if not undamaged:
        print("WARNING: no undamaged images found (paths.undamaged_dirs). VehiDE contains "
              "only damaged cars, so the classifier will not learn the 'undamaged' class.")
    if not damaged:
        print("WARNING: no extra damaged images (paths.damaged_dirs). Add whole-car photos of "
              "damaged cars, or the classifier may judge framing instead of damage.")

    val_json, val_dir = paths.get("val_json_path"), paths.get("val_img_dir")
    if val_json and val_dir and resolve_path(val_json).exists():
        train_ds = VehicleDataset(str(json_path), str(img_dir),
                                  [p for p in undamaged if not is_holdout(p)],
                                  [p for p in damaged if not is_holdout(p)])
        test_ds = VehicleDataset(str(resolve_path(val_json)), str(resolve_path(val_dir)),
                                 [p for p in undamaged if is_holdout(p)],
                                 [p for p in damaged if is_holdout(p)])
        return train_ds, test_ds

    base_dataset = VehicleDataset(str(json_path), str(img_dir), undamaged, damaged)
    train_len = int(0.8 * len(base_dataset))
    test_len = len(base_dataset) - train_len
    return random_split(base_dataset, [train_len, test_len],
                        generator=torch.Generator().manual_seed(42))


def _counts(ds):
    if hasattr(ds, "label_counts"):
        return ds.label_counts()
    labels = [ds.dataset.data[i][1] for i in ds.indices]
    return len(labels) - sum(labels), sum(labels)


def main():
    config = load_config()
    train_cfg = config.get("training") or {}
    output_model = resolve_path(config["paths"].get("output_model", "weights/resnet50_damage.pth"))
    batch_size = int(train_cfg.get("resnet_batch", 64))
    workers = int(train_cfg.get("workers", 4))
    epochs = int(train_cfg.get("resnet_epochs", 10))

    train_transform = transforms.Compose(
        [
            # Random zoom-crops show both classes at every framing, so
            # "close-up vs whole car" stops being a shortcut for the label.
            transforms.RandomResizedCrop(224, scale=(0.3, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(5.0),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.RandomGrayscale(0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_subset, test_subset = build_datasets(config)
    n_undamaged, n_damaged = _counts(train_subset)
    print(f"Train: {n_damaged} damaged / {n_undamaged} undamaged | "
          f"Test: {_counts(test_subset)[1]} damaged / {_counts(test_subset)[0]} undamaged")

    train_loader = DataLoader(
        TransformSubset(train_subset, train_transform),
        batch_size=batch_size, shuffle=True, num_workers=workers,
    )
    test_loader = DataLoader(
        TransformSubset(test_subset, test_transforms),
        batch_size=batch_size, shuffle=False, num_workers=workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device, weights="DEFAULT")

    # Weight the loss by inverse class frequency — VehiDE is far larger than
    # most undamaged-car sets, and an unweighted loss just predicts "damaged".
    total = max(n_undamaged + n_damaged, 1)
    class_weights = torch.tensor(
        [total / (2 * max(n_undamaged, 1)), total / (2 * max(n_damaged, 1))],
        dtype=torch.float, device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", 0.1, 2)

    os.makedirs(os.path.dirname(str(output_model)) or ".", exist_ok=True)

    best_test_acc = 0.0
    best_acc_diff = float("inf")

    for epoch in range(epochs):
        train_loss, train_acc = _run_epoch(model, train_loader, device, criterion, optimizer)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.3f} %")

        test_loss, test_acc = _run_epoch(model, test_loader, device, criterion)
        print(f"Epoch {epoch + 1}/{epochs} - Test Loss:  {test_loss:.5f} | Test Acc:  {test_acc:.3f} %")

        scheduler.step(test_acc)

        acc_diff = abs(train_acc - test_acc)
        is_best = test_acc > best_test_acc or (test_acc == best_test_acc and acc_diff < best_acc_diff)

        if is_best:
            best_test_acc = test_acc
            best_acc_diff = acc_diff
            torch.save(model.state_dict(), output_model)
            print(f"New best model saved to {output_model} — Test Acc: {best_test_acc:.3f} %, Diff: {best_acc_diff:.3f} %")

        print("-" * 60)


if __name__ == "__main__":
    main()