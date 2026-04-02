from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

MODEL_PATH = "models/flower_classifier.keras"
LEGACY_MODEL_PATH = "models/flower_classifier.h5"
CLASS_NAMES_PATH = "models/class_names.json"


_model = None
_class_names = None


def model_file_exists() -> bool:
    return Path(MODEL_PATH).exists() or Path(LEGACY_MODEL_PATH).exists()


def _load_artifacts():
    global _model, _class_names

    if _model is None:
        candidate_paths = [Path(MODEL_PATH), Path(LEGACY_MODEL_PATH)]
        load_error = None
        for path in candidate_paths:
            if not path.exists():
                continue
            try:
                _model = tf.keras.models.load_model(path, compile=False)
                break
            except Exception as exc:  # noqa: BLE001
                load_error = exc

        if _model is None and load_error is not None:
            raise RuntimeError(
                f"Model file exists but failed to load. Re-train to generate {MODEL_PATH}."
            ) from load_error

        if _model is None:
            raise FileNotFoundError("Model not found. Train the model first.")

    if _class_names is None:
        if not Path(CLASS_NAMES_PATH).exists():
            raise FileNotFoundError("Class names not found. Train the model first.")
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as fp:
            _class_names = json.load(fp)

    return _model, _class_names


def predict_image(image_path: str) -> dict:
    model, class_names = _load_artifacts()

    image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    array = tf.keras.utils.img_to_array(image)
    array = tf.expand_dims(array, 0)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)

    predictions = model.predict(array, verbose=0)[0]
    idx = int(np.argmax(predictions))

    return {
        "predicted_class": class_names[idx],
        "confidence": float(predictions[idx]),
        "class_probabilities": {
            cls: float(predictions[i]) for i, cls in enumerate(class_names)
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run single-image prediction")
    parser.add_argument("image_path", type=str, help="Path to image file")
    args = parser.parse_args()

    print(predict_image(args.image_path))
