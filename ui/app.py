from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from PIL import Image

API_BASE_DEFAULT = os.getenv("API_BASE", "http://localhost:8000")
PROOF_DIR = Path("docs/proofs")

st.set_page_config(page_title="VisionOps Cycle", layout="wide")

st.markdown(
    """
<style>
:root {
  --bg: #f6f7f9;
  --card: #ffffff;
  --text: #1f2937;
  --muted: #6b7280;
  --line: #e5e7eb;
  --accent: #0f766e;
}

.stApp {
  background: var(--bg);
  color: var(--text);
}

.block {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 0.9rem 1rem;
}

.small {
  color: var(--muted);
  font-size: 0.9rem;
}

.title {
  font-size: 1.5rem;
  font-weight: 700;
  margin: 0;
}

.subtitle {
  color: var(--muted);
  margin-top: 0.2rem;
}

.stTabs [data-baseweb="tab"] {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 10px;
}

.stTabs [aria-selected="true"] {
  border-color: var(--accent);
}
</style>
""",
    unsafe_allow_html=True,
)


def _safe_get_json(url: str, timeout: int = 12) -> tuple[dict | None, str | None]:
    try:
        res = requests.get(url, timeout=timeout)
        if not res.ok:
            return None, f"HTTP {res.status_code}: {res.text[:180]}"
        return res.json(), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _count_upload_queue(upload_root: Path = Path("uploads")) -> int:
    if not upload_root.exists():
        return 0
    return sum(1 for p in upload_root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def _load_logs() -> pd.DataFrame:
    log_path = Path("logs/prediction_log.csv")
    if not log_path.exists():
        return pd.DataFrame(columns=["timestamp_utc", "filename", "predicted_class", "confidence", "latency_ms"])
    df = pd.read_csv(log_path)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    return df


def _placeholder_proof(path: Path, title: str, lines: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.axis("off")
    ax.text(0.02, 0.86, title, fontsize=18, color="#111827", weight="bold", transform=ax.transAxes)
    y = 0.68
    for line in lines:
        ax.text(0.02, y, line, fontsize=12, color="#4b5563", transform=ax.transAxes)
        y -= 0.12
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _build_proof_images(last_prediction: dict | None = None) -> dict[str, Path]:
    PROOF_DIR.mkdir(parents=True, exist_ok=True)

    prediction_img = PROOF_DIR / "01_prediction_proof.png"
    train_img = PROOF_DIR / "02_train_proof.png"
    monitor_img = PROOF_DIR / "03_monitor_proof.png"
    guide_img = PROOF_DIR / "04_guide_proof.png"

    if isinstance(last_prediction, dict) and "class_probabilities" in last_prediction:
        probs = pd.DataFrame(
            {
                "class_name": list(last_prediction["class_probabilities"].keys()),
                "probability": list(last_prediction["class_probabilities"].values()),
            }
        ).sort_values("probability", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 5.2))
        ax.bar(probs["class_name"], probs["probability"], color="#0f766e")
        ax.set_title("Prediction Proof: Class Probabilities")
        ax.set_ylabel("Probability")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(prediction_img, dpi=150)
        plt.close(fig)
    else:
        _placeholder_proof(
            prediction_img,
            "Prediction Proof",
            [
                "Run one image prediction in the Predict tab.",
                "Then click 'Generate / Refresh Proof Images'.",
            ],
        )

    metrics_path = Path("models/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as fp:
            metrics = json.load(fp)
        history = metrics.get("history", {}) if isinstance(metrics, dict) else {}
        loss = history.get("loss") or []
        val_loss = history.get("val_loss") or []
        if loss and val_loss and len(loss) == len(val_loss):
            epochs = list(range(1, len(loss) + 1))
            fig, ax = plt.subplots(figsize=(10, 5.2))
            ax.plot(epochs, loss, label="Train Loss", linewidth=2.2, color="#0f766e")
            ax.plot(epochs, val_loss, label="Val Loss", linewidth=2.2, color="#ea580c")
            ax.set_title("Training Proof: Loss Curves")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            fig.tight_layout()
            fig.savefig(train_img, dpi=150)
            plt.close(fig)
        else:
            _placeholder_proof(train_img, "Training Proof", ["Training history not found in metrics file."])
    else:
        _placeholder_proof(train_img, "Training Proof", ["models/metrics.json is missing."])

    log_df = _load_logs()
    if not log_df.empty and "latency_ms" in log_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5.2))
        x_vals = log_df["timestamp_utc"] if "timestamp_utc" in log_df.columns else log_df.index
        ax.plot(x_vals, log_df["latency_ms"], color="#2563eb", linewidth=2.2)
        ax.set_title("Monitoring Proof: Inference Latency")
        ax.set_ylabel("Latency (ms)")
        fig.tight_layout()
        fig.savefig(monitor_img, dpi=150)
        plt.close(fig)
    else:
        _placeholder_proof(monitor_img, "Monitoring Proof", ["No prediction logs yet. Run prediction first."])

    _placeholder_proof(
        guide_img,
        "Guide Proof: Rubric Flow",
        [
            "1. Prediction: one image -> class/confidence/latency",
            "2. Train: upload images -> trigger retraining",
            "3. Monitor: metrics + logs + charts",
            "4. Guide: rubric mapping and evidence",
        ],
    )

    return {
        "Prediction": prediction_img,
        "Train": train_img,
        "Monitor": monitor_img,
        "Guide": guide_img,
    }


with st.sidebar:
    st.header("Setup")
    api_base = st.text_input("API URL", value=API_BASE_DEFAULT)
    if st.button("Refresh"):
        st.rerun()

    st.markdown("---")
    st.subheader("Rubric Checklist")
    st.write("1. Predict one image")
    st.write("2. Upload + retrain")
    st.write("3. Monitor metrics/logs")
    st.write("4. Show proof images")

    st.markdown("---")
    model_ready = Path("models/flower_classifier.keras").exists() or Path("models/flower_classifier.h5").exists()
    st.write(f"Model: {'Ready' if model_ready else 'Missing'}")
    st.write(f"Confusion matrix: {'Ready' if Path('models/confusion_matrix.png').exists() else 'Missing'}")
    st.write(f"Feature summary: {'Ready' if Path('models/feature_summary.csv').exists() else 'Missing'}")

health, health_err = _safe_get_json(f"{api_base.strip()}/health")
metrics, metrics_err = _safe_get_json(f"{api_base.strip()}/metrics")

if health is None or metrics is None:
    st.error(f"API not reachable: {health_err or metrics_err or 'Unknown error'}")
    st.stop()

upload_queue = _count_upload_queue()

st.markdown(
    """
<div class="block">
  <div class="title">VisionOps Cycle Dashboard</div>
  <div class="subtitle">A clean, rubric-focused app for prediction, retraining, monitoring, and evidence.</div>
</div>
""",
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Model", "Ready" if health.get("model_available") else "Missing")
k2.metric("Requests", int(metrics.get("total_requests", 0)))
k3.metric("Avg latency", f"{metrics.get('avg_latency_ms', 0) or 0:.2f} ms")
k4.metric("P95 latency", f"{metrics.get('p95_latency_ms', 0) or 0:.2f} ms")

home_tab, pred_tab, train_tab, monitor_tab, guide_tab, proofs_tab = st.tabs([
    "Dashboard",
    "Predict",
    "Train",
    "Monitor",
    "Guide",
    "Proofs",
])

with home_tab:
    st.subheader("What this app is for")
    st.write(
        "This app demonstrates the full MLOps cycle end-to-end: single-image prediction, retraining with new data, "
        "and service monitoring."
    )
    st.write("Recommended demo order: Predict -> Train -> Monitor -> Guide -> Proofs")
    st.info("To return anytime, click the Dashboard tab.")

with pred_tab:
    st.subheader("Prediction")
    st.caption("Upload one image and run a single prediction.")
    uploaded = st.file_uploader("Image file", type=["jpg", "jpeg", "png"], key="predict_uploader")

    if uploaded is not None:
        image_bytes = uploaded.read()
        c_left, c_right = st.columns([1.2, 1])
        with c_left:
            st.image(Image.open(BytesIO(image_bytes)), caption=uploaded.name, use_column_width=True)

        with c_right:
            if st.button("Run Prediction", type="primary"):
                resp = requests.post(
                    f"{api_base.strip()}/predict",
                    files={"file": (uploaded.name, image_bytes, uploaded.type or "image/jpeg")},
                    timeout=60,
                )
                if resp.ok:
                    result = resp.json()
                    st.session_state["last_prediction"] = result
                    st.success("Prediction complete")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Class", result["predicted_class"])
                    m2.metric("Confidence", f"{result['confidence']:.2%}")
                    m3.metric("Latency", f"{result['latency_ms']} ms")

                    probs = pd.DataFrame(
                        {
                            "class_name": list(result["class_probabilities"].keys()),
                            "probability": list(result["class_probabilities"].values()),
                        }
                    ).sort_values("probability", ascending=False)

                    fig = px.bar(probs, x="class_name", y="probability", title="Class Probabilities")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(probs.head(3), use_container_width=True, hide_index=True)
                else:
                    st.error(resp.text)

    if "last_prediction" in st.session_state:
        lp = st.session_state["last_prediction"]
        st.caption(f"Last result: {lp.get('predicted_class')} ({lp.get('confidence', 0):.2%})")

with train_tab:
    st.subheader("Retraining")
    st.caption("Upload files for a class, then trigger retraining.")

    class_name = st.text_input("Class name", value="new_class")
    bulk = st.file_uploader(
        "Upload training images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="train_uploader",
    )

    up_col, rt_col = st.columns(2)

    with up_col:
        if st.button("Upload Files"):
            if not bulk:
                st.warning("Please select at least one image.")
            else:
                files = [("files", (f.name, f.read(), f.type or "image/jpeg")) for f in bulk]
                res = requests.post(
                    f"{api_base.strip()}/upload-data",
                    data={"class_name": class_name},
                    files=files,
                    timeout=120,
                )
                if res.ok:
                    st.success("Upload complete")
                    st.json(res.json())
                else:
                    st.error(res.text)

        st.caption(f"Upload queue: {upload_queue}")

    with rt_col:
        epochs = st.slider("Epochs", min_value=1, max_value=10, value=3)
        if st.button("Trigger Retraining", type="primary"):
            with st.spinner("Running retraining..."):
                res = requests.post(
                    f"{api_base.strip()}/trigger-retrain",
                    params={"epochs": epochs},
                    timeout=3600,
                )
            if res.ok:
                payload = res.json()
                st.session_state["last_retrain"] = payload
                st.success("Retraining done")
                st.json(payload)
            else:
                st.error(res.text)

    if "last_retrain" in st.session_state:
        lr = st.session_state["last_retrain"]
        st.caption(f"Last retrain: {lr.get('status', 'unknown')} | moved files: {lr.get('moved_files', 'N/A')}")

with monitor_tab:
    st.subheader("Monitoring")
    st.caption("Health, metrics, logs, and model artifacts.")

    left, right = st.columns(2)
    with left:
        cm = Path("models/confusion_matrix.png")
        if cm.exists():
            st.image(str(cm), caption="Confusion Matrix", use_column_width=True)
        else:
            st.info("No confusion matrix yet.")

    with right:
        st.caption("Health payload")
        st.json(health)
        st.caption("Metrics payload")
        st.json(metrics)

    logs = _load_logs()
    if not logs.empty:
        if "latency_ms" in logs.columns:
            lat_fig = px.line(logs, x="timestamp_utc", y="latency_ms", title="Inference Latency")
            st.plotly_chart(lat_fig, use_container_width=True)

        if "predicted_class" in logs.columns:
            class_counts = logs["predicted_class"].value_counts().reset_index()
            class_counts.columns = ["class_name", "count"]
            dist_fig = px.bar(class_counts, x="class_name", y="count", title="Prediction Count by Class")
            st.plotly_chart(dist_fig, use_container_width=True)

        if "confidence" in logs.columns and not logs["confidence"].dropna().empty:
            conf_fig = px.histogram(logs, x="confidence", nbins=20, title="Confidence Distribution")
            st.plotly_chart(conf_fig, use_container_width=True)
    else:
        st.info("No logs yet. Run predictions to populate this section.")

    fs = Path("models/feature_summary.csv")
    if fs.exists():
        fdf = pd.read_csv(fs)
        w_fig = px.bar(fdf, x="class_name", y="avg_width", title="Average Width by Class")
        h_fig = px.bar(fdf, x="class_name", y="avg_height", title="Average Height by Class")
        b_fig = px.bar(fdf, x="class_name", y="avg_brightness", title="Average Brightness by Class")
        st.plotly_chart(w_fig, use_container_width=True)
        st.plotly_chart(h_fig, use_container_width=True)
        st.plotly_chart(b_fig, use_container_width=True)
    else:
        st.warning("Feature summary is missing.")

with guide_tab:
    st.subheader("Guide")
    st.markdown("1. **Prediction**: run one image and show class, confidence, latency.")
    st.markdown("2. **Retraining**: upload new data and trigger retraining.")
    st.markdown("3. **Monitoring**: show metrics, logs, latency, and class distribution.")
    st.markdown("4. **Feature interpretation**: show width, height, and brightness charts.")
    st.markdown("5. **Proof pack**: generate and export images for all sections.")
    st.info("Return to the main page anytime using the Dashboard tab.")

with proofs_tab:
    st.subheader("Proof Images")
    st.caption("Generate evidence images for Prediction, Train, Monitor, and Guide.")

    if st.button("Generate / Refresh Proof Images", type="primary"):
        proof_map = _build_proof_images(st.session_state.get("last_prediction"))
        st.session_state["proof_paths"] = {k: str(v) for k, v in proof_map.items()}
        st.success("Proof images generated in docs/proofs.")

    proof_paths = st.session_state.get("proof_paths") or {
        "Prediction": str(PROOF_DIR / "01_prediction_proof.png"),
        "Train": str(PROOF_DIR / "02_train_proof.png"),
        "Monitor": str(PROOF_DIR / "03_monitor_proof.png"),
        "Guide": str(PROOF_DIR / "04_guide_proof.png"),
    }

    for label in ["Prediction", "Train", "Monitor", "Guide"]:
        path = Path(proof_paths[label])
        st.markdown(f"### {label}")
        if path.exists():
            st.image(str(path), use_column_width=True)
            with open(path, "rb") as fp:
                st.download_button(
                    label=f"Download {label} proof",
                    data=fp,
                    file_name=path.name,
                    mime="image/png",
                    key=f"download_{label.lower()}_proof",
                )
        else:
            st.warning(f"Missing: {path}")

    st.info("Return to the Dashboard tab when you finish downloading proofs.")
