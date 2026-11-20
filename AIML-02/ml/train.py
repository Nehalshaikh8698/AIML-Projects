"""Train a DenseNet-121 multi-label classifier on NIH ChestX-ray14.

This is an educational baseline. It is not a clinically validated model.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, densenet121
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural Thickening", "Hernia",
]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ChestXrayDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_dir: Path, training: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        if training:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image_path = self.image_dir / row["Image Index"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        target = torch.tensor([float(row[label]) for label in LABELS], dtype=torch.float32)
        return self.transform(image), target


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata(csv_path: Path, image_dir: Path, limit: int | None) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"NIH labels CSV was not found: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory was not found: {image_dir}")
    source = pd.read_csv(csv_path)
    required = {"Image Index", "Finding Labels"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    records = []
    for row in source[["Image Index", "Finding Labels"]].itertuples(index=False):
        image_name, finding_text = row
        image_path = image_dir / str(image_name)
        if not image_path.is_file():
            continue
        found = set(str(finding_text).split("|"))
        record = {"Image Index": str(image_name)}
        for label in LABELS:
            record[label] = int(label in found)
        records.append(record)
        if limit and len(records) >= limit:
            break
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No matching image files were found. Check your extracted images folder.")
    return frame


def macro_auc(targets: np.ndarray, probabilities: np.ndarray) -> tuple[float | None, dict[str, float | None]]:
    per_label: dict[str, float | None] = {}
    values = []
    for index, label in enumerate(LABELS):
        y = targets[:, index]
        if len(np.unique(y)) < 2:
            per_label[label] = None
            continue
        try:
            score = float(roc_auc_score(y, probabilities[:, index]))
            per_label[label] = score
            values.append(score)
        except ValueError:
            per_label[label] = None
    return (float(np.mean(values)) if values else None), per_label


def run_epoch(model, loader, criterion, optimizer, device, training: bool) -> tuple[float, np.ndarray, np.ndarray]:
    model.train(training)
    total_loss = 0.0
    all_targets, all_probs = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in tqdm(loader, leave=False):
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().cpu().item()) * images.size(0)
            all_targets.append(targets.detach().cpu().numpy())
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return total_loss / len(loader.dataset), np.concatenate(all_targets), np.concatenate(all_probs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DenseNet121 on NIH ChestX-ray14")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/NIH-ChestXray14/images")
    parser.add_argument("--labels-csv", type=Path, default=ROOT / "data/NIH-ChestXray14/Data_Entry_2017_v2020.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for a quick pipeline smoke test")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download ImageNet initialization weights")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    frame = load_metadata(args.labels_csv, args.data_dir, args.limit)
    # NIH image names start with a patient identifier (for example 00000001_000.png).
    # Keep all images from a patient in a single split to reduce patient leakage.
    groups = frame["Image Index"].str.split("_").str[0]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed)
    train_indices, remainder_indices = next(splitter.split(frame, groups=groups))
    train_frame = frame.iloc[train_indices].reset_index(drop=True)
    remainder = frame.iloc[remainder_indices].reset_index(drop=True)
    remainder_groups = groups.iloc[remainder_indices].reset_index(drop=True)
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=args.seed)
    val_indices, test_indices = next(splitter_val.split(remainder, groups=remainder_groups))
    val_frame = remainder.iloc[val_indices].reset_index(drop=True)
    test_frame = remainder.iloc[test_indices].reset_index(drop=True)
    print(f"Images: train={len(train_frame)}, validation={len(val_frame)}, test={len(test_frame)}")
    print("Patient-grouped split based on NIH image filename patient identifiers.")

    train_ds = ChestXrayDataset(train_frame, args.data_dir, training=True)
    val_ds = ChestXrayDataset(val_frame, args.data_dir)
    test_ds = ChestXrayDataset(test_frame, args.data_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    weights = None if args.no_pretrained else DenseNet121_Weights.DEFAULT
    model = densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
    model.to(device)

    y_train = train_frame[LABELS].to_numpy(dtype=np.float32)
    positives = y_train.sum(axis=0)
    negatives = len(y_train) - positives
    pos_weight = np.divide(negatives, np.maximum(positives, 1.0))
    pos_weight = np.clip(pos_weight, 1.0, 20.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    checkpoint_path = ROOT / "backend/app/models/densenet121_nih14.pth"
    metrics_path = ROOT / "ml/outputs/metrics.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss, val_targets, val_probs = run_epoch(model, val_loader, criterion, optimizer, device, training=False)
        val_auc, val_per_label = macro_auc(val_targets, val_probs)
        monitor = val_auc if val_auc is not None else -val_loss
        scheduler.step(monitor)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | macro_val_auc={val_auc}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "macro_val_auc": val_auc, "per_label_val_auc": val_per_label})
        if monitor > best_auc:
            best_auc = monitor
            torch.save({
                "model_state_dict": model.state_dict(),
                "labels": LABELS,
                "threshold": 0.50,
                "architecture": "torchvision-densenet121",
                "training_dataset": "NIH ChestX-ray14",
                "seed": args.seed,
                "epoch": epoch,
            }, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_targets, test_probs = run_epoch(model, test_loader, criterion, optimizer, device, training=False)
    test_auc, test_per_label = macro_auc(test_targets, test_probs)
    report = {
        "dataset": "NIH ChestX-ray14",
        "architecture": "DenseNet121",
        "labels": LABELS,
        "seed": args.seed,
        "image_counts": {"train": len(train_ds), "validation": len(val_ds), "test": len(test_ds)},
        "history": history,
        "test_loss": test_loss,
        "macro_test_auc": test_auc,
        "per_label_test_auc": test_per_label,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "warning": "Educational baseline only. Patient grouping uses filename prefixes and still requires independent validation; not for clinical use.",
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Test macro AUROC: {test_auc}")


if __name__ == "__main__":
    main()
