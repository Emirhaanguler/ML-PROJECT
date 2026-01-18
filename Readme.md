# TUANDROMD Malware Detection — ML Term Project

This repository contains an end-to-end machine learning pipeline for **binary classification (malware vs. goodware)** using the **TUANDROMD** dataset from the UCI Machine Learning Repository.

Dataset link (UCI): https://archive.ics.uci.edu/dataset/855/tuandromd+%28tezpur+university+android+malware+dataset%29  
You can access and download the dataset from this page.

## Dataset
- Source: UCI Machine Learning Repository — TUANDROMD
- Task: Binary classification (malware vs. goodware)
- Size: 4,465 instances, 241 features (+ label)

## Project Contents
- `TUANDROMD_ML.ipynb` — EDA, preprocessing, model training, tuning, and evaluation
- `TUANDROMD.csv` — dataset file (if included)
- `requirements.txt` — dependencies

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
