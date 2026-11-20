"""Validate the NIH ChestX-ray14 folder and write a small dataset inventory."""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/NIH-ChestXray14"))
    parser.add_argument("--output", type=Path, default=Path("ml/outputs/dataset_inventory.json"))
    args = parser.parse_args()
    csv_path = args.data_dir / "Data_Entry_2017.csv"
    image_dir = args.data_dir / "images"
    if not csv_path.exists() or not image_dir.exists():
        raise SystemExit("Expected Data_Entry_2017.csv and images/ under the NIH dataset directory.")
    frame = pd.read_csv(csv_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "rows_in_csv": int(len(frame)),
        "images_present": sum(1 for name in frame["Image Index"] if (image_dir / name).is_file()),
        "finding_label_examples": frame["Finding Labels"].value_counts().head(20).to_dict(),
    }, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
