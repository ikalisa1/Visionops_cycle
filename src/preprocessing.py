from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageStat

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def _class_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [d for d in path.iterdir() if d.is_dir()]


def _has_images(path: Path) -> bool:
    if not path.exists():
        return False
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            return True
    return False


def _has_direct_images(path: Path) -> bool:
    if not path.exists():
        return False
    return any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in path.iterdir())


def _resolve_dataset_source_dir(path: Path) -> Path:
    """Resolve to the folder that directly contains class directories with images."""
    class_dirs = _class_dirs(path)
    if any(_has_direct_images(d) for d in class_dirs):
        return path

    # TensorFlow flower dataset can extract as <root>/flower_photos/<class_dirs>.
    if len(class_dirs) == 1:
        nested = class_dirs[0]
        nested_class_dirs = _class_dirs(nested)
        if any(_has_direct_images(d) for d in nested_class_dirs):
            return nested

    return path


def acquire_flower_dataset(base_data_dir: str = "data") -> tuple[Path, Path]:
    """Download and split a non-tabular image dataset if local data is absent."""
    data_root = Path(base_data_dir)
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    if _class_dirs(train_dir) and _class_dirs(test_dir) and _has_images(train_dir) and _has_images(test_dir):
        return train_dir, test_dir

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale class folders from previous incomplete runs.
    for folder in _class_dirs(train_dir):
        shutil.rmtree(folder, ignore_errors=True)
    for folder in _class_dirs(test_dir):
        shutil.rmtree(folder, ignore_errors=True)

    dataset_zip = tf.keras.utils.get_file(
        "flower_photos",
        origin="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
        untar=True,
    )
    source_dir = _resolve_dataset_source_dir(Path(dataset_zip))

    random.seed(42)
    for class_dir in _class_dirs(source_dir):
        images = list(class_dir.glob("*.jpg"))
        random.shuffle(images)

        split_idx = int(0.8 * len(images))
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        class_train = train_dir / class_dir.name
        class_test = test_dir / class_dir.name
        class_train.mkdir(parents=True, exist_ok=True)
        class_test.mkdir(parents=True, exist_ok=True)

        for img in train_images:
            shutil.copy2(img, class_train / img.name)

        for img in test_images:
            shutil.copy2(img, class_test / img.name)

    return train_dir, test_dir


def create_datasets(base_data_dir: str = "data"):
    train_dir, test_dir = acquire_flower_dataset(base_data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_names = train_ds.class_names

    train_ds = train_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, test_ds, class_names


def summarize_dataset_features(train_dir: str = "data/train", output_csv: str = "models/feature_summary.csv"):
    """Create simple interpretable image-level feature summaries by class."""
    rows = []
    train_path = Path(train_dir)

    for class_dir in _class_dirs(train_path):
        widths = []
        heights = []
        brightness = []

        for img_path in class_dir.glob("*.jpg"):
            with Image.open(img_path) as img:
                rgb_img = img.convert("RGB")
                w, h = rgb_img.size
                stat = ImageStat.Stat(rgb_img)
                mean_brightness = float(np.mean(stat.mean))

                widths.append(w)
                heights.append(h)
                brightness.append(mean_brightness)

        if not widths:
            continue

        rows.append(
            {
                "class_name": class_dir.name,
                "image_count": len(widths),
                "avg_width": float(np.mean(widths)),
                "avg_height": float(np.mean(heights)),
                "avg_brightness": float(np.mean(brightness)),
            }
        )

    df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def save_class_names(class_names: list[str], output_path: str = "models/class_names.json"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(class_names, fp, indent=2)
