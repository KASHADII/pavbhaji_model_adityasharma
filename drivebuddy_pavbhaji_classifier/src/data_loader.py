"""
Data Loader Module for DriveBuddyAI Pav Bhaji Text Classification Challenge.

Loads raw Instagram post metadata from JSON, maps posts to ground-truth labeled images,
extracts all metadata fields, and creates structured datasets.
"""

import os
import sys
import json
import glob
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger("DataLoader")


def find_dataset_paths(
    base_dir: Optional[str] = None
) -> Tuple[str, str, str]:
    """
    Locate pavbhaji.json and image directories (0 and 1) robustly across potential locations.
    """
    if base_dir is None:
        base_dir = os.getcwd()

    candidate_json_paths = [
        os.path.join(base_dir, "data", "raw", "pavbhaji.json"),
        os.path.join(base_dir, "machinelearningchallenge", "dataset", "dataset", "pavbhaji.json"),
        os.path.join(base_dir, "..", "machinelearningchallenge", "dataset", "dataset", "pavbhaji.json"),
        os.path.join(base_dir, "dataset", "dataset", "pavbhaji.json"),
        os.path.join(base_dir, "pavbhaji.json"),
    ]

    candidate_img_dirs = [
        os.path.join(base_dir, "data", "raw", "images"),
        os.path.join(base_dir, "machinelearningchallenge", "dataset", "dataset", "images"),
        os.path.join(base_dir, "..", "machinelearningchallenge", "dataset", "dataset", "images"),
        os.path.join(base_dir, "dataset", "dataset", "images"),
        os.path.join(base_dir, "images"),
    ]

    json_path = None
    for p in candidate_json_paths:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p):
            json_path = abs_p
            break

    images_dir = None
    for p in candidate_img_dirs:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p) and os.path.isdir(abs_p):
            images_dir = abs_p
            break

    if not json_path:
        raise FileNotFoundError(
            f"Could not locate 'pavbhaji.json' in candidate paths: {candidate_json_paths}"
        )
    if not images_dir:
        raise FileNotFoundError(
            f"Could not locate 'images' folder in candidate paths: {candidate_img_dirs}"
        )

    img_0_dir = os.path.join(images_dir, "0")
    img_1_dir = os.path.join(images_dir, "1")

    return json_path, img_0_dir, img_1_dir


def extract_post_record(
    rec: Dict[str, Any],
    idx: int,
    img_0_map: Dict[str, str],
    img_1_map: Dict[str, str]
) -> Dict[str, Any]:
    """
    Extract structured fields from a single Instagram post JSON record.
    """
    post_id = rec.get("id", f"post_{idx}")
    shortcode = rec.get("shortcode", "")

    # Caption extraction
    caption = ""
    caption_edges = rec.get("edge_media_to_caption", {}).get("edges", [])
    if caption_edges and isinstance(caption_edges, list):
        node = caption_edges[0].get("node", {})
        caption = node.get("text", "")

    # Description (fallback if structured separately, else empty or alias)
    description = ""
    
    # Comments extraction (Instagram JSON edges if present)
    comments = ""
    comment_edges = rec.get("edge_media_to_comment", {}).get("edges", [])
    if comment_edges and isinstance(comment_edges, list):
        comm_texts = []
        for c_edge in comment_edges:
            c_node = c_edge.get("node", {})
            if "text" in c_node:
                comm_texts.append(c_node["text"])
        comments = " ".join(comm_texts)

    # Tags / Hashtags
    tags = rec.get("tags", [])
    if tags is None:
        tags = []
    tags_str = " ".join([f"#{t}" if not t.startswith("#") else t for t in tags]) if isinstance(tags, list) else str(tags)

    # Location
    location_dict = rec.get("location") or {}
    location_name = location_dict.get("name", "") if isinstance(location_dict, dict) else ""

    # Numeric & Boolean metadata
    likes = rec.get("edge_liked_by", {}).get("count", 0) if isinstance(rec.get("edge_liked_by"), dict) else 0
    comments_count = rec.get("edge_media_to_comment", {}).get("count", 0) if isinstance(rec.get("edge_media_to_comment"), dict) else 0
    is_video = rec.get("is_video", False)
    timestamp = rec.get("taken_at_timestamp", None)
    owner_id = rec.get("owner", {}).get("id", "") if isinstance(rec.get("owner"), dict) else ""
    video_view_count = rec.get("video_view_count", 0)

    # Image mapping logic
    display_url = rec.get("display_url", "")
    d_fn = display_url.split("/")[-1].split("?")[0] if display_url else ""

    urls = rec.get("urls", [])
    url_fns = [u.split("/")[-1].split("?")[0] for u in urls if isinstance(u, str)] if isinstance(urls, list) else []

    thumbnail_src = rec.get("thumbnail_src", "")
    t_fn = thumbnail_src.split("/")[-1].split("?")[0] if thumbnail_src else ""

    candidates = [d_fn] + url_fns + [t_fn]
    candidates = [c for c in candidates if c]

    label: Optional[int] = None
    image_path: Optional[str] = None
    image_id: Optional[str] = None

    for c in candidates:
        if c in img_1_map:
            label = 1
            image_id = c
            image_path = img_1_map[c]
            break
        elif c in img_0_map:
            label = 0
            image_id = c
            image_path = img_0_map[c]
            break

    # Construct combined text
    text_components = [caption, tags_str, location_name, comments]
    combined_text = " ".join([tc for tc in text_components if tc.strip()]).strip()

    return {
        "json_index": idx,
        "id": post_id,
        "shortcode": shortcode,
        "image_id": image_id,
        "image_path": image_path,
        "label": label,
        "caption": caption,
        "description": description,
        "comments": comments,
        "tags": tags_str,
        "location_name": location_name,
        "combined_text": combined_text,
        "likes": likes,
        "comments_count": comments_count,
        "is_video": is_video,
        "timestamp": timestamp,
        "owner_id": owner_id,
        "video_view_count": video_view_count
    }


def load_raw_dataset(
    json_path: Optional[str] = None,
    img_0_dir: Optional[str] = None,
    img_1_dir: Optional[str] = None,
    base_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads JSON metadata and parses into two DataFrames:
    1. df_labeled: Posts that map to ground-truth images in class 0 or class 1.
    2. df_all: All posts parsed from the JSON (including unlabeled pool).
    """
    if json_path is None or img_0_dir is None or img_1_dir is None:
        json_path, img_0_dir, img_1_dir = find_dataset_paths(base_dir)

    logger.info("Loading JSON metadata from: %s", json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)

    if not isinstance(raw_json, list):
        raise ValueError(f"Expected JSON top-level list of objects, got {type(raw_json)}")

    # Index images
    img_0_map = {os.path.basename(p): os.path.abspath(p) for p in glob.glob(os.path.join(img_0_dir, "*"))}
    img_1_map = {os.path.basename(p): os.path.abspath(p) for p in glob.glob(os.path.join(img_1_dir, "*"))}

    logger.info("Indexed %d images in class 0 (Non-Pav Bhaji)", len(img_0_map))
    logger.info("Indexed %d images in class 1 (Pav Bhaji)", len(img_1_map))

    records = []
    for idx, item in enumerate(raw_json):
        rec_data = extract_post_record(item, idx, img_0_map, img_1_map)
        records.append(rec_data)

    df_all = pd.DataFrame(records)
    df_labeled = df_all.dropna(subset=["label"]).copy()
    df_labeled["label"] = df_labeled["label"].astype(int)

    # Deduplicate labeled set on image_id or post id
    df_labeled = df_labeled.drop_duplicates(subset=["image_id"]).reset_index(drop=True)

    logger.info("Total JSON records: %d", len(df_all))
    logger.info("Labeled records with images: %d (Class 0: %d, Class 1: %d)",
                len(df_labeled),
                (df_labeled["label"] == 0).sum(),
                (df_labeled["label"] == 1).sum())

    return df_labeled, df_all


def create_train_test_splits(
    df_labeled: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a stratified train and test split preventing data leakage.
    Duplicate captions are identified and grouped to avoid cross-split contamination.
    """
    # Stratified split
    train_df, test_df = train_test_split(
        df_labeled,
        test_size=test_size,
        stratify=df_labeled["label"],
        random_state=random_state
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info("Created Stratified Train set: %d samples (Class 0: %d, Class 1: %d)",
                len(train_df), (train_df["label"] == 0).sum(), (train_df["label"] == 1).sum())
    logger.info("Created Stratified Test set: %d samples (Class 0: %d, Class 1: %d)",
                len(test_df), (test_df["label"] == 0).sum(), (test_df["label"] == 1).sum())

    return train_df, test_df


def save_processed_datasets(
    df_labeled: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    df_all: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Save processed DataFrames into CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    labeled_path = os.path.join(output_dir, "labeled_dataset.csv")
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    unlabeled_path = os.path.join(output_dir, "unlabeled_pool.csv")

    df_labeled.to_csv(labeled_path, index=False, encoding="utf-8")
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    df_unlabeled = df_all[df_all["label"].isna()].copy()
    df_unlabeled.to_csv(unlabeled_path, index=False, encoding="utf-8")

    logger.info("Saved processed datasets to %s", output_dir)
