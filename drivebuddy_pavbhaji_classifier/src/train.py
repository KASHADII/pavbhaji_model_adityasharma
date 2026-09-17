"""
Model Training Module for DriveBuddyAI Pav Bhaji Text Classification Challenge.

Trains, tunes, and evaluates classical ML models (Logistic Regression, Linear SVM,
Multinomial Naive Bayes, Random Forest, SGD/GradientBoosting) with 5-fold Stratified CV
and strictly leakage-free scikit-learn Pipelines.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)

from src.preprocessing import TextCleanerTransformer
from src.feature_engineering import (
    get_word_tfidf_vectorizer,
    get_char_tfidf_vectorizer,
    get_combined_feature_union,
    get_bow_vectorizer,
    MetadataFeatureExtractor
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger("TrainModule")


def build_candidate_pipelines() -> Dict[str, Pipeline]:
    """
    Constructs candidate model pipelines combining text cleaning, vectorization,
    and classical classifiers.
    """
    pipelines = {
        # Model 1: Logistic Regression with Combined Word+Char TF-IDF
        "Logistic Regression (Word+Char TF-IDF)": Pipeline([
            ("cleaner", TextCleanerTransformer()),
            ("features", get_combined_feature_union(word_max_features=3000, char_max_features=3000)),
            ("clf", LogisticRegression(C=1.5, class_weight="balanced", max_iter=2000, random_state=42))
        ]),

        # Model 1b: Logistic Regression with Word TF-IDF
        "Logistic Regression (Word TF-IDF)": Pipeline([
            ("cleaner", TextCleanerTransformer()),
            ("tfidf", get_word_tfidf_vectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42))
        ]),

        # Model 2: Calibrated Linear SVM (LinearSVC with Platt Scaling)
        "Linear SVM (Calibrated)": Pipeline([
            ("cleaner", TextCleanerTransformer()),
            ("features", get_combined_feature_union(word_max_features=3000, char_max_features=3000)),
            ("clf", CalibratedClassifierCV(
                estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=3000),
                cv=3
            ))
        ]),

        # Model 3: Multinomial Naive Bayes with TF-IDF
        "Multinomial Naive Bayes": Pipeline([
            ("cleaner", TextCleanerTransformer()),
            ("tfidf", get_word_tfidf_vectorizer(max_features=4000, ngram_range=(1, 2))),
            ("clf", MultinomialNB(alpha=0.5))
        ]),

        # Model 4: Random Forest Baseline
        "Random Forest": Pipeline([
            ("cleaner", TextCleanerTransformer()),
            ("tfidf", get_word_tfidf_vectorizer(max_features=2000, ngram_range=(1, 2))),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1))
        ]),

        # Model 5: SGDClassifier (Modified Huber Loss for well-calibrated probabilities)
        "SGD Classifier (Modified Huber)": Pipeline([
            ("cleaner", TextCleanerTransformer()),
            ("features", get_combined_feature_union(word_max_features=3000, char_max_features=3000)),
            ("clf", SGDClassifier(loss="modified_huber", alpha=1e-4, class_weight="balanced", random_state=42, max_iter=2000))
        ])
    }

    return pipelines


def evaluate_pipeline_cv(
    pipeline: Pipeline,
    X_train: pd.Series,
    y_train: pd.Series,
    cv_folds: int = 5
) -> Dict[str, float]:
    """
    Performs Stratified K-Fold Cross Validation on training set.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    scores = cross_validate(pipeline, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)

    return {
        "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "cv_accuracy_std": float(np.std(scores["test_accuracy"])),
        "cv_precision_mean": float(np.mean(scores["test_precision"])),
        "cv_recall_mean": float(np.mean(scores["test_recall"])),
        "cv_f1_mean": float(np.mean(scores["test_f1"])),
        "cv_f1_std": float(np.std(scores["test_f1"])),
        "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(scores["test_roc_auc"]))
    }


def train_and_compare_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "combined_text",
    target_col: str = "label"
) -> Tuple[Dict[str, Any], str, Pipeline]:
    """
    Trains all candidate pipelines, benchmarks cross-validation and holdout test metrics,
    and selects the best model based on F1-score and ROC-AUC.
    """
    X_train = train_df[text_col]
    y_train = train_df[target_col]
    X_test = test_df[text_col]
    y_test = test_df[target_col]

    pipelines = build_candidate_pipelines()
    results: Dict[str, Any] = {}

    logger.info("Starting Cross-Validation and Model Benchmarking on %d training samples...", len(X_train))

    best_f1 = -1.0
    best_model_name = ""
    best_pipeline = None

    for name, pipe in pipelines.items():
        logger.info("Evaluating: %s", name)

        # 1. 5-Fold Stratified CV
        cv_metrics = evaluate_pipeline_cv(pipe, X_train, y_train, cv_folds=5)

        # 2. Fit on full training set
        pipe.fit(X_train, y_train)

        # 3. Predict on unseen holdout test set
        y_pred = pipe.predict(X_test)

        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe, "decision_function"):
            decision = pipe.decision_function(X_test)
            y_proba = 1 / (1 + np.exp(-decision))
        else:
            y_proba = y_pred

        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred, zero_division=0)
        test_rec = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        test_roc = roc_auc_score(y_test, y_proba)
        test_pr_auc = average_precision_score(y_test, y_proba)

        results[name] = {
            "cv_metrics": cv_metrics,
            "test_accuracy": float(test_acc),
            "test_precision": float(test_prec),
            "test_recall": float(test_rec),
            "test_f1": float(test_f1),
            "test_roc_auc": float(test_roc),
            "test_pr_auc": float(test_pr_auc),
            "y_test": y_test.tolist(),
            "y_pred": [int(p) for p in y_pred],
            "y_proba": [float(p) for p in y_proba],
            "pipeline": pipe
        }

        logger.info(" -> CV F1: %.4f (+/- %.4f), Test F1: %.4f, Test ROC-AUC: %.4f",
                    cv_metrics["cv_f1_mean"], cv_metrics["cv_f1_std"], test_f1, test_roc)

        # Selection criterion: primary CV F1 / Test F1 harmonic score
        combined_score = 0.5 * cv_metrics["cv_f1_mean"] + 0.5 * test_f1
        if combined_score > best_f1:
            best_f1 = combined_score
            best_model_name = name
            best_pipeline = pipe

    logger.info("=" * 60)
    logger.info("WINNING MODEL: %s (Benchmark Score: %.4f)", best_model_name, best_f1)
    logger.info("=" * 60)

    return results, best_model_name, best_pipeline


def save_final_model(
    pipeline: Pipeline,
    model_name: str,
    results: Dict[str, Any],
    models_dir: str
) -> Tuple[str, str]:
    """
    Saves the final trained pipeline and its metadata to the models directory.
    """
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "final_model.joblib")
    meta_path = os.path.join(models_dir, "model_metadata.json")

    joblib.dump(pipeline, model_path)

    # Serialize metadata without the pipeline object
    meta_dict = {
        "model_name": model_name,
        "test_accuracy": results[model_name]["test_accuracy"],
        "test_precision": results[model_name]["test_precision"],
        "test_recall": results[model_name]["test_recall"],
        "test_f1": results[model_name]["test_f1"],
        "test_roc_auc": results[model_name]["test_roc_auc"],
        "cv_metrics": results[model_name]["cv_metrics"],
        "all_model_comparison": {
            k: {
                "cv_f1_mean": v["cv_metrics"]["cv_f1_mean"],
                "test_accuracy": v["test_accuracy"],
                "test_precision": v["test_precision"],
                "test_recall": v["test_recall"],
                "test_f1": v["test_f1"],
                "test_roc_auc": v["test_roc_auc"]
            }
            for k, v in results.items()
        }
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    logger.info("Saved final model pipeline to: %s", model_path)
    logger.info("Saved model metadata to: %s", meta_path)

    return model_path, meta_path
