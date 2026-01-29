# TUANDROMD Malware Detection — ML Term Project

This repository contains an end-to-end machine learning pipeline for **Android malware detection** using the **TUANDROMD** dataset from the UCI Machine Learning Repository.  
The task is formulated as a **binary classification problem (malware vs. goodware)** based on static permission- and API-level features.

---

## Dataset

- **Source:** UCI Machine Learning Repository — TUANDROMD  
- **Dataset link:**  
  https://archive.ics.uci.edu/dataset/855/tuandromd+%28tezpur+university+android+malware+dataset%29
- **Task:** Binary classification (malware vs. goodware)
- **Size:** 4,465 instances, 241 features (+ label)
- **Feature type:** Binary indicators (0/1) representing permission and API usage

---

## Project Contents

- **TUANDROMD_ML.ipynb**  
  Exploratory data analysis (EDA), preprocessing, feature selection, model training, hyperparameter tuning, and evaluation
- **TUANDROMD.csv**  
  Dataset file (if included)
- **requirements.txt**  
  Python dependencies

---

## Methodology & Evaluation

- Preprocessing and feature selection are performed **within a pipeline** to prevent data leakage.
- Feature selection includes:
  - Low-variance feature removal
  - Correlation-based filtering
  - Mutual Information–based selection
- Model selection is based on **5-fold Stratified Cross-Validation** using **weighted F1-score**.
- Evaluated models:
  - Dummy Classifier (baseline)
  - Logistic Regression
  - Random Forest
  - HistGradientBoosting
- The **hold-out test set** is used **only once** for final evaluation of the selected model.

### Evaluation Metrics
- Weighted F1-score
- Accuracy
- ROC-AUC
- Precision–Recall AUC
- Confusion Matrix

---

## Results 

Ensemble-based models outperform linear baselines, demonstrating the importance of capturing non-linear feature interactions.  
The selected model achieves strong performance on the hold-out test set, confirming that static permission- and API-based features are effective for Android malware detection.

---

## Reproducibility

Random seeds are fixed where applicable to ensure reproducibility of results.

---

## Setup

**Python 3.10+** is recommended.

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
