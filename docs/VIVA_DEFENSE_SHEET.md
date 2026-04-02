# Viva Defense Sheet (One Page)

## 1. Problem Statement
- Task: Build an end-to-end machine learning pipeline on non-tabular data with deployment, monitoring, retraining, and load testing.
- Chosen modality: image classification.

## 2. Why This Model?
- Used transfer learning with MobileNetV2 because:
  - strong baseline for image tasks,
  - faster convergence on limited compute,
  - suitable for deployment-focused workflows.

## 3. Optimization Techniques Used
- Pretrained backbone (MobileNetV2)
- Data augmentation (flip/rotation/zoom)
- Dropout regularization
- L2 regularization on final layer
- Adam optimizer
- EarlyStopping (restore best weights)
- ReduceLROnPlateau

## 4. Evaluation Metrics Reported
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- Training loss
- Validation loss

## 5. Retraining Design
- User uploads bulk images through API/UI.
- Files are saved in uploads path and logged to SQLite (`uploaded_files`).
- Uploaded images are preprocessed before merging to training data.
- Trigger endpoint starts retraining and updates model artifacts.

## 6. Prediction Design
- Single image prediction endpoint (`POST /predict`).
- Returns predicted class + confidence + class probability distribution.
- Inference latency logged to CSV and SQLite for monitoring.

## 7. Monitoring and Insights
- Uptime and model availability via `/health`.
- Request volume and latency (avg + p95) via `/metrics`.
- Feature insights shown in UI:
  - average width by class,
  - average height by class,
  - average brightness by class,
  - latency trend over time.

## 8. Deployment and Scalability
- Dockerized API and UI.
- Locust used for flood/load simulation.
- Experiment design: compare 1 vs 2 vs 4 API containers under same user load.

## 9. Typical Examiner Questions and Short Answers
1. "How do you prove retraining happened?"
- New metrics/model artifacts are generated, and uploaded data status in SQLite progresses through upload/preprocess/retrained.

2. "How do you prove prediction correctness?"
- Notebook prediction demo compares predicted class against known true label from test folder structure.

3. "Why macro metrics?"
- Macro averaging treats classes equally and is robust for potential class imbalance.

4. "What production risks remain?"
- Need stronger auth, versioned model registry, canary rollout, and automated drift detection for enterprise scale.

## 10. Future Improvements
- Add CI/CD pipeline for automated tests and deployment.
- Add data drift and concept drift monitoring.
- Add model versioning with rollback support.
- Add asynchronous retraining jobs with queue workers.
