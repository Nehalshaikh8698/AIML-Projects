"""Train a MONAI 3D U-Net on Medical Segmentation Decathlon Task06_Lung.

Expected folder: Task06_Lung/dataset.json, imagesTr/, labelsTr/.
This is an educational baseline and is not clinically validated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped,
    LoadImaged, Orientationd, RandAffined, RandCropByPosNegLabeld,
    RandFlipd, ScaleIntensityRanged, Spacingd,
)
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def build_model(device):
    return UNet(
        spatial_dims=3, in_channels=1, out_channels=2,
        channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2,
    ).to(device)


def main():
    parser = argparse.ArgumentParser(description="Train MONAI 3D U-Net on MSD Task06_Lung")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/Task06_Lung")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-train-cases", type=int, default=None, help="Optional small subset for a pipeline smoke test")
    args = parser.parse_args()

    if not (args.data_dir / "dataset.json").exists():
        raise SystemExit(f"Missing {args.data_dir / 'dataset.json'}. Download and extract MSD Task06_Lung first.")
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")

    training_files = load_decathlon_datalist(
        str(args.data_dir / "dataset.json"), is_segmentation=True,
        data_list_key="training", base_dir=str(args.data_dir),
    )
    import random
    random.Random(42).shuffle(training_files)
    if args.max_train_cases:
        training_files = training_files[:args.max_train_cases]
    if len(training_files) < 2:
        raise SystemExit("At least two training image/label pairs are required.")
    split = max(1, int(len(training_files) * 0.8))
    train_files, val_files = training_files[:split], training_files[split:]
    if not val_files:
        val_files = train_files[-1:]
        train_files = train_files[:-1]

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=2),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandAffined(keys=["image", "label"], prob=0.25, rotate_range=(0.05, 0.05, 0.05), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"]),
    ])
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])

    train_ds = CacheDataset(train_files, train_transforms, cache_rate=0.05, num_workers=args.workers)
    val_ds = CacheDataset(val_files, val_transforms, cache_rate=0.05, num_workers=args.workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training cases: {len(train_files)} | validation cases: {len(val_files)} | device: {device}")
    print("Task06_Lung segmentation labels are used as provided by the MSD dataset; verify label definitions before interpreting the masks.")
    model = build_model(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    metric = DiceMetric(include_background=False, reduction="mean")

    best_dice = -1.0
    checkpoint_path = ROOT / "backend/app/models/unet3d_msd_lung.pth"
    metrics_path = ROOT / "ml/outputs/segmentation_metrics.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            steps += 1

        model.eval()
        metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device).long()
                logits = sliding_window_inference(images, (96, 96, 96), 1, model, overlap=0.25)
                prediction = torch.argmax(logits, dim=1)
                prediction_1h = torch.nn.functional.one_hot(prediction, num_classes=2).movedim(-1, 1).float()
                labels_1h = torch.nn.functional.one_hot(labels[:, 0].long(), num_classes=2).movedim(-1, 1).float()
                metric(y_pred=prediction_1h, y=labels_1h)
        val_dice = float(metric.aggregate().item())
        metric.reset()
        train_loss = total_loss / max(steps, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_dice": val_dice})
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, validation_dice={val_dice:.4f}")
        if np.isfinite(val_dice) and val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": "monai-3d-unet",
                "dataset": "Medical Segmentation Decathlon Task06_Lung",
                "validation_dice": best_dice,
                "epoch": epoch,
            }, checkpoint_path)

    report = {
        "dataset": "Medical Segmentation Decathlon Task06_Lung",
        "architecture": "MONAI 3D U-Net",
        "train_cases": len(train_files),
        "validation_cases": len(val_files),
        "history": history,
        "best_validation_dice": best_dice if best_dice >= 0 else None,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "warning": "Educational baseline only. Dataset split and results require independent review; not for clinical use.",
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
