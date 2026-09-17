"""
Evaluation and Diagnostics Module for DriveBuddyAI Pav Bhaji Text Classification Challenge.

Generates evaluation metrics, confusion matrices, ROC/PR curves, error analysis,
and publication-quality visualization figures.
"""

import os
import re
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.feature_extraction.text import CountVectorizer

# Configure plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["figure.dpi"] = 300


def plot_class_distribution(df_labeled: pd.DataFrame, figures_dir: str) -> str:
    """
    Plots the ground-truth class distribution.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    counts = df_labeled["label"].value_counts().sort_index()
    labels = ["Non-Pav Bhaji (0)", "Pav Bhaji (1)"]
    colors = ["#4A90E2", "#FF6B6B"]

    bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 5, f"{int(h)} ({h/len(df_labeled):.1%})",
                ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_ylim(0, max(counts.values) + 40)
    ax.set_ylabel("Number of Samples", fontweight="bold")
    ax.set_title("Ground-Truth Class Distribution (Labeled Dataset)", pad=15)
    plt.tight_layout()

    out_path = os.path.join(figures_dir, "class_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_missing_values(df_all: pd.DataFrame, figures_dir: str) -> str:
    """
    Plots the missing value percentage across all raw metadata fields.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    missing = df_all.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        ax.text(0.5, 0.5, "No Missing Values in Primary Metadata", ha="center", va="center", fontsize=12)
    else:
        bars = ax.barh(missing.index, missing.values, color="#E67E22", edgecolor="black", linewidth=1.1)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}%",
                    ha="left", va="center", fontweight="bold", fontsize=10)
        ax.set_xlim(0, 110)
        ax.set_xlabel("Missing Percentage (%)", fontweight="bold")
        ax.set_title("Missing Values Across Metadata Fields (1500 Total Records)", pad=15)

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "missing_values_analysis.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_text_length_distribution(df_labeled: pd.DataFrame, figures_dir: str) -> str:
    """
    Plots text character and word length distributions partitioned by class.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    df = df_labeled.copy()
    df["char_len"] = df["combined_text"].astype(str).str.len()
    df["word_len"] = df["combined_text"].astype(str).str.split().str.len()
    df["Class"] = df["label"].map({0: "Non-Pav Bhaji (0)", 1: "Pav Bhaji (1)"})

    # Character Length
    sns.boxplot(x="Class", y="char_len", hue="Class", data=df, ax=axes[0], palette=["#4A90E2", "#FF6B6B"], legend=False)
    axes[0].set_ylabel("Character Count", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_title("Character Length Distribution by Class")

    # Word Length
    sns.boxplot(x="Class", y="word_len", hue="Class", data=df, ax=axes[1], palette=["#4A90E2", "#FF6B6B"], legend=False)
    axes[1].set_ylabel("Word Count", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_title("Word Count Distribution by Class")

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "text_length_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_top_words_by_class(df_labeled: pd.DataFrame, figures_dir: str, top_n: int = 15) -> str:
    """
    Plots top informative word tokens for Class 0 vs Class 1.
    """
    from src.preprocessing import clean_text

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for idx, target_label in enumerate([0, 1]):
        sub_df = df_labeled[df_labeled["label"] == target_label]
        cleaned_corpus = [clean_text(t) for t in sub_df["combined_text"]]

        # Stopwords tailored for Instagram text to expose discriminative food terms
        custom_stops = [
            "the", "and", "to", "of", "in", "for", "is", "on", "that", "by", "this",
            "with", "i", "you", "it", "not", "or", "be", "are", "from", "at", "as",
            "your", "all", "have", "my", "more", "we", "so", "our", "me", "if", "an",
            "food", "foodie", "foodblogger", "foodphotography", "instafood", "foodies",
            "follow", "tag", "like", "post", "us", "repost", "dm", "link", "bio"
        ]

        vec = CountVectorizer(stop_words=custom_stops, max_features=top_n, ngram_range=(1, 1))
        counts = vec.fit_transform(cleaned_corpus)
        word_freq = pd.Series(np.asarray(counts.sum(axis=0)).ravel(), index=vec.get_feature_names_out()).sort_values(ascending=True)

        color = "#4A90E2" if target_label == 0 else "#FF6B6B"
        title = "Top Words in Non-Pav Bhaji Posts (0)" if target_label == 0 else "Top Words in Pav Bhaji Posts (1)"

        axes[idx].barh(word_freq.index, word_freq.values, color=color, edgecolor="black", linewidth=0.9)
        axes[idx].set_title(title)
        axes[idx].set_xlabel("Frequency in Class Corpus", fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "top_words_by_class.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_top_hashtags_by_class(df_labeled: pd.DataFrame, figures_dir: str, top_n: int = 15) -> str:
    """
    Extracts and compares top hashtags by class.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for idx, target_label in enumerate([0, 1]):
        sub_df = df_labeled[df_labeled["label"] == target_label]
        all_tags = []
        for raw_text in sub_df["combined_text"].astype(str):
            tags = re.findall(r"#(\w+)", raw_text.lower())
            all_tags.extend(tags)

        tag_series = pd.Series(all_tags).value_counts().head(top_n).sort_values(ascending=True)
        color = "#4A90E2" if target_label == 0 else "#FF6B6B"
        title = f"Top Hashtags: {'Non-Pav Bhaji (0)' if target_label == 0 else 'Pav Bhaji (1)'}"

        axes[idx].barh(["#" + t for t in tag_series.index], tag_series.values, color=color, edgecolor="black", linewidth=0.9)
        axes[idx].set_title(title)
        axes[idx].set_xlabel("Occurrences", fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "top_hashtags_by_class.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_confusion_matrix_figure(y_true: List[int], y_pred: List[int], model_name: str, figures_dir: str) -> str:
    """
    Plots an annotated confusion matrix with True Positives, True Negatives,
    False Positives, and False Negatives clearly labeled.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    annot_matrix = np.array([
        [f"TN\n{tn}", f"FP\n{fp}"],
        [f"FN\n{fn}", f"TP\n{tp}"]
    ])

    sns.heatmap(cm, annot=annot_matrix, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred: Not Pav Bhaji (0)", "Pred: Pav Bhaji (1)"],
                yticklabels=["True: Not Pav Bhaji (0)", "True: Pav Bhaji (1)"],
                annot_kws={"fontsize": 14, "fontweight": "bold"})

    ax.set_title(f"Confusion Matrix - {model_name}", pad=15)
    plt.tight_layout()

    out_path = os.path.join(figures_dir, "confusion_matrix.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_model_comparison(results: Dict[str, Any], figures_dir: str) -> str:
    """
    Plots a multi-metric bar chart comparing all trained models across Accuracy, F1, and ROC-AUC.
    """
    model_names = list(results.keys())
    metrics_df = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": res["test_accuracy"],
            "Precision": res["test_precision"],
            "Recall": res["test_recall"],
            "F1-Score": res["test_f1"],
            "ROC-AUC": res["test_roc_auc"]
        }
        for name, res in results.items()
    ])

    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_melted = pd.melt(metrics_df, id_vars=["Model"], value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                             var_name="Metric", value_name="Score")

    sns.barplot(x="Model", y="Score", hue="Metric", data=metrics_melted, ax=ax, palette="Set2", edgecolor="black", linewidth=0.8)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Score", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_title("Model Performance Benchmark on Holdout Test Set", pad=15)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "model_performance_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_roc_curves(results: Dict[str, Any], figures_dir: str) -> str:
    """
    Plots ROC curves for all models on holdout test set.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for name, res in results.items():
        y_true = res["y_test"]
        y_proba = res["y_proba"]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_score:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", pad=15)
    ax.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "roc_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_precision_recall_curves(results: Dict[str, Any], figures_dir: str) -> str:
    """
    Plots Precision-Recall curves for all models on holdout test set.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for name, res in results.items():
        y_true = res["y_test"]
        y_proba = res["y_proba"]
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        pr_score = average_precision_score(y_true, y_proba)
        ax.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {pr_score:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontweight="bold")
    ax.set_ylabel("Precision", fontweight="bold")
    ax.set_title("Precision-Recall (PR) Curves", pad=15)
    ax.legend(loc="lower left", fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "precision_recall_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def perform_error_analysis(
    test_df: pd.DataFrame,
    best_model_results: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Performs deep qualitative inspection of False Positives and False Negatives.
    """
    y_test = np.array(best_model_results["y_test"])
    y_pred = np.array(best_model_results["y_pred"])
    y_proba = np.array(best_model_results["y_proba"])

    df = test_df.copy()
    df["y_true"] = y_test
    df["y_pred"] = y_pred
    df["y_proba"] = y_proba

    false_positives = df[(df["y_true"] == 0) & (df["y_pred"] == 1)]
    false_negatives = df[(df["y_true"] == 1) & (df["y_pred"] == 0)]

    fp_list = []
    for _, row in false_positives.iterrows():
        fp_list.append({
            "image_id": row.get("image_id", ""),
            "true_label": 0,
            "predicted_label": 1,
            "confidence": float(row["y_proba"]),
            "caption": str(row.get("caption", ""))[:300],
            "tags": str(row.get("tags", ""))[:150]
        })

    fn_list = []
    for _, row in false_negatives.iterrows():
        fn_list.append({
            "image_id": row.get("image_id", ""),
            "true_label": 1,
            "predicted_label": 0,
            "confidence": float(row["y_proba"]),
            "caption": str(row.get("caption", ""))[:300],
            "tags": str(row.get("tags", ""))[:150]
        })

    return {
        "false_positives": fp_list,
        "false_negatives": fn_list
    }


def generate_all_evaluation_figures(
    df_labeled: pd.DataFrame,
    df_all: pd.DataFrame,
    test_df: pd.DataFrame,
    results: Dict[str, Any],
    best_model_name: str,
    reports_dir: str
) -> Dict[str, str]:
    """
    Generates and saves all 9 visual figures.
    """
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    generated = {}
    generated["class_distribution"] = plot_class_distribution(df_labeled, figures_dir)
    generated["missing_values"] = plot_missing_values(df_all, figures_dir)
    generated["text_length"] = plot_text_length_distribution(df_labeled, figures_dir)
    generated["top_words"] = plot_top_words_by_class(df_labeled, figures_dir)
    generated["top_hashtags"] = plot_top_hashtags_by_class(df_labeled, figures_dir)
    generated["confusion_matrix"] = plot_confusion_matrix_figure(
        results[best_model_name]["y_test"],
        results[best_model_name]["y_pred"],
        best_model_name,
        figures_dir
    )
    generated["model_comparison"] = plot_model_comparison(results, figures_dir)
    generated["roc_curves"] = plot_roc_curves(results, figures_dir)
    generated["precision_recall_curves"] = plot_precision_recall_curves(results, figures_dir)

    return generated
