"""
Prediction System Module for DriveBuddyAI Pav Bhaji Text Classification Challenge.

Supports single-post JSON inference, batch JSON prediction, and batch CSV prediction.
"""

import os
import json
import logging
from typing import Dict, Any, List, Union, Optional
import pandas as pd
import numpy as np
import joblib

from src.preprocessing import clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger("PredictSystem")

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_model.joblib")


def load_trained_model(model_path: Optional[str] = None):
    """
    Loads the trained scikit-learn pipeline from disk.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    if not os.path.exists(model_path):
        # Check parent models directory if called from different CWD
        alt_path = os.path.abspath(os.path.join("models", "final_model.joblib"))
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Model file not found at '{model_path}'. Please train the model first.")

    model = joblib.load(model_path)
    return model


def format_single_record_text(metadata: Dict[str, Any]) -> str:
    """
    Extracts and combines text fields from a single Instagram post JSON record.
    """
    # 1. Caption
    caption = ""
    caption_edges = metadata.get("edge_media_to_caption", {}).get("edges", [])
    if caption_edges and isinstance(caption_edges, list):
        node = caption_edges[0].get("node", {})
        caption = node.get("text", "")
    elif "caption" in metadata and isinstance(metadata["caption"], str):
        caption = metadata["caption"]

    # 2. Description
    description = metadata.get("description", "")

    # 3. Comments
    comments = ""
    comment_edges = metadata.get("edge_media_to_comment", {}).get("edges", [])
    if comment_edges and isinstance(comment_edges, list):
        comm_texts = []
        for c_edge in comment_edges:
            c_node = c_edge.get("node", {})
            if "text" in c_node:
                comm_texts.append(c_node["text"])
        comments = " ".join(comm_texts)
    elif "comments" in metadata and isinstance(metadata["comments"], str):
        comments = metadata["comments"]

    # 4. Tags
    tags = metadata.get("tags", [])
    if tags is None:
        tags = []
    tags_str = " ".join([f"#{t}" if not t.startswith("#") else t for t in tags]) if isinstance(tags, list) else str(tags)

    # 5. Location
    location_dict = metadata.get("location") or {}
    location_name = location_dict.get("name", "") if isinstance(location_dict, dict) else str(location_dict)

    combined = " ".join([c for c in [caption, description, tags_str, location_name, comments] if c.strip()]).strip()
    return combined


def predict_post(
    metadata: Union[Dict[str, Any], str],
    model: Optional[Any] = None,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Predicts whether a single Instagram post corresponds to Pav Bhaji or Not Pav Bhaji.

    Parameters
    ----------
    metadata : dict or JSON string
        The Instagram post metadata record.
    model : Pipeline, optional
        Pre-loaded sklearn pipeline. If None, loads from model_path.
    model_path : str, optional
        Path to saved final_model.joblib.

    Returns
    -------
    dict:
        image_id: str
        prediction: int (0 or 1)
        prediction_label: str ("Pav Bhaji" or "Not Pav Bhaji")
        confidence_or_score: float (model estimated probability of Pav Bhaji)
        explanation: str
    """
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    if model is None:
        model = load_trained_model(model_path)

    # Determine image_id
    image_id = metadata.get("image_id")
    if not image_id:
        d_url = metadata.get("display_url", "")
        image_id = d_url.split("/")[-1].split("?")[0] if d_url else metadata.get("id", "unknown_post")

    combined_text = format_single_record_text(metadata)
    text_series = pd.Series([combined_text])

    pred = int(model.predict(text_series)[0])

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(text_series)[0][1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(text_series)[0])
        prob = float(1.0 / (1.0 + np.exp(-score)))
    else:
        prob = float(pred)

    pred_label = "Pav Bhaji" if pred == 1 else "Not Pav Bhaji"

    return {
        "image_id": image_id,
        "prediction": pred,
        "prediction_label": pred_label,
        "confidence_or_score": round(prob, 4),
        "explanation": (
            f"Estimated probability of Pav Bhaji: {prob:.2%}. "
            "Note: Probability is a model-estimated score from text/metadata features, not a vision verification."
        )
    }


def predict_batch_json(
    json_path: str,
    output_path: Optional[str] = None,
    model_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Runs prediction on a JSON file containing a list of Instagram post metadata records.
    """
    model = load_trained_model(model_path)
    with open(json_path, "r", encoding="utf-8") as f:
        posts = json.load(f)

    if not isinstance(posts, list):
        posts = [posts]

    results = []
    for post in posts:
        res = predict_post(post, model=model)
        results.append(res)

    df_preds = pd.DataFrame(results)
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df_preds.to_csv(output_path, index=False, encoding="utf-8")
        logger.info("Saved batch JSON predictions to %s", output_path)

    return df_preds


def predict_batch_csv(
    csv_path: str,
    text_column: str = "combined_text",
    output_path: Optional[str] = None,
    model_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Runs prediction on a CSV file containing Instagram posts.
    """
    model = load_trained_model(model_path)
    df = pd.read_csv(csv_path)

    if text_column not in df.columns:
        # Try constructing from caption, tags, comments if present
        cols = [c for c in ["caption", "description", "tags", "location_name", "comments"] if c in df.columns]
        if cols:
            df["combined_text"] = df[cols].fillna("").agg(" ".join, axis=1)
            text_column = "combined_text"
        else:
            raise ValueError(f"CSV must contain '{text_column}' or text metadata columns.")

    preds = model.predict(df[text_column])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df[text_column])[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(df[text_column])
        probs = 1.0 / (1.0 + np.exp(-scores))
    else:
        probs = preds

    df_out = pd.DataFrame({
        "image_id": df.get("image_id", df.get("id", [f"item_{i}" for i in range(len(df))])),
        "prediction": preds.astype(int),
        "prediction_label": ["Pav Bhaji" if p == 1 else "Not Pav Bhaji" for p in preds],
        "confidence_or_score": np.round(probs, 4)
    })

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df_out.to_csv(output_path, index=False, encoding="utf-8")
        logger.info("Saved batch CSV predictions to %s", output_path)

    return df_out
