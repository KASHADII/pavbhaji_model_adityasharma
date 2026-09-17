"""
Feature Engineering Module for DriveBuddyAI Pav Bhaji Text Classification Challenge.

Builds text vectorization pipelines (Word TF-IDF, Char TF-IDF, FeatureUnion, BoW)
and dense metadata/lexical feature extractors.
"""

import re
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, RobustScaler

# Competing food keywords commonly found in non-pav bhaji Instagram posts
COMPETING_FOOD_KEYWORDS = [
    "chicken", "tikka", "biryani", "burger", "pizza", "pasta", "pakoda", "pakora",
    "dosa", "idli", "thali", "thalipeeth", "chaat", "roll", "paneer", "maggi",
    "momos", "shawarma", "noodles", "sandwich", "fries", "waffle", "icecream",
    "dal", "makhani", "kabab", "kebab", "vada", "samosa", "paratha", "chole",
    "bhature", "golgappa", "pani", "puri", "bhelpuri", "sevpuri", "frankie"
]

# Signature Pav Bhaji related keywords
PAV_BHAJI_SIGNATURE_KEYWORDS = [
    "pav", "bhaji", "bhajji", "pao", "bhaaji", "amul", "butter", "cheese",
    "tawa", "masala", "makhan", "limbu", "lemon", "onion", "kanda", "sardar",
    "cannon", "maruti", "khaugalli", "extra butter", "amul butter", "cheesepavbhaji"
]

# General culinary & restaurant context keywords
CULINARY_KEYWORDS = [
    "delicious", "yummy", "taste", "tasty", "streetfood", "stall", "recipe",
    "eating", "foodie", "foodblogger", "spicy", "hot", "piping", "serve",
    "plate", "best", "famous", "order", "hunger", "bhukkad", "craving"
]


class MetadataFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts dense lexical, statistical, and domain-specific metadata features from raw text.
    """

    def __init__(self):
        self.feature_names_: List[str] = []

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        """
        Extracts numeric features from an array or Series of text.
        """
        if isinstance(X, pd.Series):
            texts = X.tolist()
        elif isinstance(X, np.ndarray):
            texts = X.tolist()
        elif isinstance(X, list):
            texts = X
        else:
            texts = [str(X)]

        features_list = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text) if text is not None else ""

            text_lower = text.lower()
            words = text_lower.split()
            word_count = len(words)
            char_count = len(text)

            # Mention & Hashtag counts
            mention_count = len(re.findall(r"@[\w\.\-]+", text))
            hashtag_count = len(re.findall(r"#\w+", text))

            # Pav Bhaji specific keyword matches
            pb_hits = sum(1 for kw in PAV_BHAJI_SIGNATURE_KEYWORDS if kw in text_lower)
            comp_hits = sum(1 for kw in COMPETING_FOOD_KEYWORDS if kw in text_lower)
            culinary_hits = sum(1 for kw in CULINARY_KEYWORDS if kw in text_lower)

            # Focus ratio: degree to which text focuses on Pav Bhaji vs competing dishes
            focus_ratio = (pb_hits + 1.0) / (comp_hits + pb_hits + 1.0)
            comp_density = comp_hits / (word_count + 1.0)
            pb_density = pb_hits / (word_count + 1.0)

            features_list.append([
                char_count,
                word_count,
                mention_count,
                hashtag_count,
                pb_hits,
                comp_hits,
                culinary_hits,
                focus_ratio,
                comp_density,
                pb_density
            ])

        self.feature_names_ = [
            "char_count",
            "word_count",
            "mention_count",
            "hashtag_count",
            "pavbhaji_keyword_hits",
            "competing_food_hits",
            "culinary_hits",
            "pavbhaji_focus_ratio",
            "competing_food_density",
            "pavbhaji_keyword_density"
        ]

        return np.array(features_list, dtype=np.float32)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_


def get_word_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2
) -> TfidfVectorizer:
    """
    Returns standard word-level TF-IDF vectorizer with sublinear scaling.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
        strip_accents="unicode"
    )


def get_char_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (3, 5),
    min_df: int = 2
) -> TfidfVectorizer:
    """
    Returns character-level TF-IDF vectorizer to capture sub-word, typos, and Hinglish morphemes.
    """
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
        strip_accents="unicode"
    )


def get_combined_feature_union(
    word_max_features: int = 4000,
    char_max_features: int = 4000
) -> FeatureUnion:
    """
    Combines Word TF-IDF and Character TF-IDF in parallel into a single sparse matrix.
    """
    return FeatureUnion([
        ("word_tfidf", get_word_tfidf_vectorizer(max_features=word_max_features, ngram_range=(1, 2))),
        ("char_tfidf", get_char_tfidf_vectorizer(max_features=char_max_features, ngram_range=(3, 5)))
    ])


def get_bow_vectorizer(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2
) -> CountVectorizer:
    """
    Returns Bag-of-Words CountVectorizer baseline.
    """
    return CountVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        strip_accents="unicode"
    )
