# DriveBuddyAI Pav Bhaji Text Classification System

> A production-grade text classification machine learning system to predict whether an Instagram post image features **Pav Bhaji** (Class 1) or **Not Pav Bhaji** (Class 0) strictly using post metadata without computer vision.

---

## 📌 Project Overview & Challenge Objective

In social media content moderation and food discovery platforms, indexing dishes from user-generated captions and tags is challenging.
The goal of this challenge is to train a machine learning model that predicts whether an Instagram post features **Pav Bhaji** using only the metadata stored in `pavbhaji.json`.

### Key Constraints:
- **Text-Based Classification**: Strictly uses text and metadata fields (caption, tags, location, comments, engagement statistics).
- **Zero Computer Vision**: No CNNs, vision transformers, image pixels, or visual embeddings are used.
- **Leakage Prevention**: All posts originate from `#pavbhaji` search queries, so naive keyword matching fails. The model must learn deep discriminative patterns (dish descriptions, co-occurring food words, focus ratios).

---

## 📊 Dataset Summary

| Metric / Attribute | Value | Description |
| :--- | :--- | :--- |
| **Total JSON Records** | `1,500` | Raw Instagram post records in `pavbhaji.json` |
| **Labeled Ground-Truth Images** | `452` | Verified images in `dataset/images/` |
| **Class 0 (Non-Pav Bhaji)** | `269` (59.5%) | Posts featuring Chicken Tikka, Pasta, Dosa, Pakoda, etc. |
| **Class 1 (Pav Bhaji)** | `183` (40.5%) | Posts featuring authentic Pav Bhaji |
| **Unlabeled Candidate Pool** | `1,048` | Additional JSON posts for future semi-supervised learning |
| **Train / Test Split** | `361` / `91` | Stratified 80/20 train/test split (`random_state=42`) |

---

## 🏆 Model Benchmarking & Results

All models were evaluated with **5-Fold Stratified Cross-Validation** on the training split and verified on the holdout test set:

| Model Architecture | 5-Fold CV F1-Score | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Word+Char TF-IDF)** ⭐ | **0.6469 ± 0.0389** | **0.6703** | **0.5800** | **0.7838** | **0.6591** | **0.6952** |
| **Logistic Regression (Word TF-IDF)** | 0.6237 ± 0.0394 | 0.6374 | 0.5333 | 0.8649 | 0.6598 | 0.6772 |
| **Linear SVM (Calibrated)** | 0.5705 ± 0.0459 | 0.6044 | 0.5135 | 0.5135 | 0.5135 | 0.6942 |
| **Multinomial Naive Bayes** | 0.5942 ± 0.0463 | 0.6154 | 0.5116 | 0.5946 | 0.6667 | 0.6637 |
| **Random Forest Baseline** | 0.6147 ± 0.0843 | 0.5934 | 0.5000 | 0.7297 | 0.5918 | 0.6532 |
| **SGD Classifier (Modified Huber)** | 0.6152 ± 0.0467 | 0.5824 | 0.4865 | 0.4865 | 0.4865 | 0.5681 |

**Winning Model**: `Logistic Regression (Word+Char TF-IDF)` achieves the best balance of F1-score (`0.6591`), high recall on Pav Bhaji instances (`78.38%`), and highest ROC-AUC (`0.6952`).

---

## 📁 Project Structure

```text
drivebuddy_pavbhaji_classifier/
│
├── data/
│   ├── raw/
│   │   ├── pavbhaji.json
│   │   └── images/
│   └── processed/
│       ├── labeled_dataset.csv
│       ├── train.csv
│       ├── test.csv
│       └── unlabeled_pool.csv
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # JSON ingestion & image mapping
│   ├── preprocessing.py        # Text normalization & hashtag decomposition
│   ├── feature_engineering.py  # Word/Char TF-IDF & metadata extractors
│   ├── train.py                # 5-fold CV & model training
│   ├── evaluate.py             # 9 diagnostic figures & error analysis
│   └── predict.py              # Single-post, batch JSON & CSV prediction
│
├── models/
│   ├── final_model.joblib      # Serialized scikit-learn pipeline
│   └── model_metadata.json     # Saved metrics and training configuration
│
├── reports/
│   ├── data_analysis_report.md # Full 13-section technical report
│   └── figures/                # 9 publication-quality diagnostic plots
│       ├── class_distribution.png
│       ├── missing_values_analysis.png
│       ├── text_length_distribution.png
│       ├── top_words_by_class.png
│       ├── top_hashtags_by_class.png
│       ├── confusion_matrix.png
│       ├── model_performance_comparison.png
│       ├── roc_curves.png
│       └── precision_recall_curves.png
│
├── predictions/
│   └── sample_predictions.csv  # Test set inference demonstration
│
├── requirements.txt
├── README.md
└── main.py
```

---

## 🚀 Quickstart & Execution

### 1. Installation
Ensure Python 3.10+ is installed:

```bash
# Clone the repository
git clone https://github.com/drivebuddyai/machinelearningchallenge.git
cd machinelearningchallenge

# Create virtual environment and install requirements
python -m venv .venv
.venv\Scripts\activate  # On Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
Runs data loading, preprocessing, 5-fold cross validation, plot generation, report authoring, and test predictions:

```bash
python main.py
```

### 3. Dedicated Execution Commands

#### Run Training & Evaluation Only:
```bash
python main.py --mode train
```

#### Run Batch Prediction on a CSV File:
```bash
python main.py --mode predict --input data/processed/test.csv --output predictions/my_preds.csv
```

#### Run Batch Prediction on a JSON File:
```bash
python main.py --mode predict --input path/to/posts.json --output predictions/json_preds.csv
```

### 4. Programmatic Python Prediction API
You can use `predict_post` directly in Python:

```python
from src.predict import predict_post

sample_post = {
    "id": "post_123",
    "edge_media_to_caption": {
        "edges": [{"node": {"text": "Craving spicy buttery Pav Bhaji with extra butter! #pavbhaji"}}]
    },
    "tags": ["pavbhaji", "streetfood", "mumbaifoodie"],
    "location": {"name": "Juhu Beach, Mumbai"}
}

result = predict_post(sample_post)
print(result)
# Output:
# {
#   'image_id': 'post_123',
#   'prediction': 1,
#   'prediction_label': 'Pav Bhaji',
#   'confidence_or_score': 0.7854,
#   'explanation': 'Estimated probability of Pav Bhaji: 78.54%...'
# }
```

---

## 📈 Diagnostic Plots & Artifacts

All plots are automatically generated at 300 DPI in `reports/figures/`:
- **Class Distribution**: [class_distribution.png](reports/figures/class_distribution.png)
- **Confusion Matrix**: [confusion_matrix.png](reports/figures/confusion_matrix.png)
- **Model Comparison**: [model_performance_comparison.png](reports/figures/model_performance_comparison.png)
- **ROC Curves**: [roc_curves.png](reports/figures/roc_curves.png)
- **Precision-Recall Curves**: [precision_recall_curves.png](reports/figures/precision_recall_curves.png)
- **Top Words & Hashtags**: [top_words_by_class.png](reports/figures/top_words_by_class.png), [top_hashtags_by_class.png](reports/figures/top_hashtags_by_class.png)

---

## 📄 Deliverables Checklist

- [x] Complete Modular Source Code (`src/data_loader.py`, `src/preprocessing.py`, `src/feature_engineering.py`, `src/train.py`, `src/evaluate.py`, `src/predict.py`)
- [x] End-to-End Orchestrator (`main.py`)
- [x] Comprehensive 13-Section Markdown Report (`reports/data_analysis_report.md`)
- [x] 9 Diagnostic Visualization Figures (`reports/figures/`)
- [x] Interactive Exploratory Jupyter Notebook (`notebooks/exploratory_analysis.ipynb`)
- [x] Saved Production Model (`models/final_model.joblib`)
- [x] Inference System with Single/Batch Support (`predictions/sample_predictions.csv`)
- [x] Dependency Manifest (`requirements.txt`)
- [x] Complete Documentation (`README.md`)
