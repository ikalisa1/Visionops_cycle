from __future__ import annotations

import csv
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.database import init_db, log_prediction, log_uploaded_file
from src.prediction import CLASS_NAMES_PATH, model_file_exists, predict_image
from src.retrain import trigger_retraining

app = FastAPI(title="Image Classification ML API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TS = time.time()
LOG_PATH = Path("logs/prediction_log.csv")
UPLOAD_ROOT = Path("uploads")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
init_db()

if not LOG_PATH.exists():
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["timestamp_utc", "filename", "predicted_class", "confidence", "latency_ms"])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TS, 2),
        "model_available": model_file_exists() and Path(CLASS_NAMES_PATH).exists(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=400, detail="Only JPG/PNG images are supported.")

    start = time.perf_counter()

    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    try:
        result = predict_image(temp_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        now_utc = datetime.now(timezone.utc).isoformat()
        writer.writerow(
            [
                now_utc,
                file.filename,
                result["predicted_class"],
                round(result["confidence"], 6),
                latency_ms,
            ]
        )

    log_prediction(
        file_name=file.filename,
        predicted_class=result["predicted_class"],
        confidence=float(result["confidence"]),
        latency_ms=float(latency_ms),
        predicted_at=now_utc,
    )

    return {**result, "latency_ms": latency_ms}


@app.post("/upload-data")
async def upload_data(class_name: str = Form(...), files: List[UploadFile] = File(...)):
    if not class_name.strip():
        raise HTTPException(status_code=400, detail="class_name is required")

    target_dir = UPLOAD_ROOT / class_name.strip().lower().replace(" ", "_")
    target_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for upload in files:
        ext = Path(upload.filename).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png"}:
            continue

        out_path = target_dir / upload.filename
        with open(out_path, "wb") as fp:
            fp.write(await upload.read())

        log_uploaded_file(
            file_name=upload.filename,
            file_path=str(out_path),
            class_name=class_name,
            status="uploaded",
            uploaded_at=datetime.now(timezone.utc).isoformat(),
        )
        saved += 1

    return {
        "status": "ok",
        "class_name": class_name,
        "saved_files": saved,
        "upload_path": str(target_dir),
    }


@app.post("/trigger-retrain")
def trigger_retrain_endpoint(epochs: int = 3):
    result = trigger_retraining(epochs=epochs)
    return result


@app.get("/metrics")
def metrics():
    if not LOG_PATH.exists():
        return {"total_requests": 0, "avg_latency_ms": None, "p95_latency_ms": None}

    df = pd.read_csv(LOG_PATH)
    if df.empty:
        return {"total_requests": 0, "avg_latency_ms": None, "p95_latency_ms": None}

    return {
        "total_requests": int(len(df)),
        "avg_latency_ms": float(df["latency_ms"].mean()),
        "p95_latency_ms": float(df["latency_ms"].quantile(0.95)),
        "top_predicted_classes": df["predicted_class"].value_counts().head(5).to_dict(),
    }
