# 3-Minute Presentation Script

## 0:00 - 0:20 | Opening
Hi everyone. This project is called VisionOps Cycle. It is an end-to-end MLOps workflow for flower image classification. The system covers model training, single-image prediction, monitoring, and retraining with newly uploaded data.

## 0:20 - 0:55 | What I Built
The notebook handles data preparation, training, and evaluation. The API serves predictions and retraining endpoints. The dashboard gives one place to run prediction, upload retraining data, monitor latency, and download evidence images for rubric submission.

## 0:55 - 1:35 | Training + Evaluation (Notebook)
Now I open the notebook and run the key cells.

I show data acquisition and preprocessing first, then model training with optimization methods:
1. Transfer learning with MobileNetV2
2. Data augmentation
3. Dropout and L2 regularization
4. Adam optimizer with learning-rate scheduling and early stopping

Then I show evaluation outputs: accuracy, precision, recall, F1-score, and train/validation loss.

## 1:35 - 2:00 | Prediction Correctness
Next, I run one prediction on a test image.

I show:
1. The selected image path
2. The true label from folder name
3. The predicted label and confidence
4. Whether the prediction matches the true label

This demonstrates single data-point prediction and correctness checking.

## 2:00 - 2:35 | Dashboard Flow
Now I switch to the dashboard.

I use the tabs in this order:
1. Predict: upload one image and run inference
2. Train: upload new class images and trigger retraining
3. Monitor: review health, latency, and prediction trends
4. Guide: show rubric mapping
5. Proofs: generate and download evidence images

This makes the demo structured and easy to follow.

## 2:35 - 2:55 | Retraining + Monitoring Value
When retraining is triggered, new uploads are merged into training data and the model is refreshed. Logs and metrics update, so we can track latency behavior and confidence distribution after changes.

## 2:55 - 3:00 | Closing
In summary, this project demonstrates the complete lifecycle: train, predict, monitor, and retrain, with clear proof artifacts ready for submission.

---

## Quick Delivery Tips
1. Speak calmly and keep eye contact after each section.
2. Do not read every line; use this as cue points.
3. If a step takes long, say: "This is already precomputed; I will show the result now."