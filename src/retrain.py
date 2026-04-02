from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

from src.database import mark_uploaded_file_preprocessed, mark_uploaded_file_retrained
from src.model import train_and_evaluate


def merge_uploaded_data(upload_dir: str = "uploads", train_dir: str = "data/train") -> int:
    upload_path = Path(upload_dir)
    train_path = Path(train_dir)
    train_path.mkdir(parents=True, exist_ok=True)

    moved = 0
    for class_dir in upload_path.iterdir() if upload_path.exists() else []:
        if not class_dir.is_dir():
            continue

        target_dir = train_path / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        for file_path in class_dir.iterdir():
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            # Preprocess uploaded image: force RGB and standardize extension for training.
            with Image.open(file_path) as img:
                rgb = img.convert("RGB")
                processed_source = file_path.with_suffix(".jpg")
                rgb.save(processed_source, format="JPEG", quality=95)

            mark_uploaded_file_preprocessed(str(file_path))

            target_path = target_dir / processed_source.name
            stem = target_path.stem
            suffix = target_path.suffix
            counter = 1
            while target_path.exists():
                target_path = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            shutil.copy2(processed_source, target_path)
            mark_uploaded_file_retrained(str(file_path))
            moved += 1

    return moved


def trigger_retraining(epochs: int = 3) -> dict:
    moved = merge_uploaded_data()
    if moved == 0:
        return {
            "status": "skipped",
            "message": "No uploaded images were found to retrain.",
            "moved_files": 0,
        }

    metrics = train_and_evaluate(epochs=epochs)
    return {
        "status": "success",
        "message": "Model retraining completed.",
        "moved_files": moved,
        "metrics": metrics,
    }
