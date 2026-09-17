"""
Text Preprocessing Module for DriveBuddyAI Pav Bhaji Text Classification Challenge.

Provides comprehensive cleaning, hashtag decomposition, emoji normalization,
and Hinglish/multilingual-friendly tokenization.
"""

import re
import unicodedata
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Regular expressions for text cleaning
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@[\w\.\-]+", re.IGNORECASE)
HASHTAG_PATTERN = re.compile(r"#(\w+)", re.IGNORECASE)
REPEATED_PUNCT_PATTERN = re.compile(r"[\.!\?,_\-~]{2,}")
WHITESPACE_PATTERN = re.compile(r"\s+")
HTML_PATTERN = re.compile(r"<.*?>")
NUMERIC_ONLY_PATTERN = re.compile(r"^\d+$")

# Compound food & location hashtag word-splitting map
COMMON_HASHTAG_SPLITS = {
    "cheesepavbhaji": "cheese pav bhaji",
    "butterpavbhaji": "butter pav bhaji",
    "mumbaipavbhaji": "mumbai pav bhaji",
    "mumbaifoodie": "mumbai foodie",
    "delhifoodie": "delhi foodie",
    "streetfood": "street food",
    "streetfoodindia": "street food india",
    "indianfood": "indian food",
    "indianstreetfood": "indian street food",
    "chickentikka": "chicken tikka",
    "butterchicken": "butter chicken",
    "breadpakoda": "bread pakoda",
    "foodporn": "food porn",
    "foodgasm": "food gasm",
    "foodphotography": "food photography",
    "foodblogger": "food blogger",
    "foodtalkindia": "food talk india",
    "desifood": "desi food",
    "comfortfood": "comfort food",
    "foodiesofinstagram": "foodies of instagram",
    "instafood": "insta food",
    "foodie": "foodie",
    "nagpurfoodie": "nagpur foodie",
    "punefoodie": "pune foodie",
    "suratfoodies": "surat foodies",
    "ahmedabadfoodie": "ahmedabad foodie",
}


def clean_text(
    text: Optional[Union[str, float]],
    remove_urls: bool = True,
    remove_mentions: bool = True,
    decompose_hashtags: bool = True,
    preserve_emojis: bool = True,
    lowercase: bool = True
) -> str:
    """
    Cleans raw Instagram text into a normalized, classification-ready representation.

    Parameters
    ----------
    text : str or None
        Raw text (caption, comments, tags, etc.)
    remove_urls : bool
        Whether to strip HTTP/HTTPS URLs.
    remove_mentions : bool
        Whether to strip @usernames.
    decompose_hashtags : bool
        Whether to split compound hashtags into individual constituent words.
    preserve_emojis : bool
        Whether to normalize emojis so they retain semantic separation.
    lowercase : bool
        Whether to convert text to lowercase.

    Returns
    -------
    str : Cleaned text string.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""

    text = str(text)

    # 1. Unicode normalization (NFKD / NFC)
    text = unicodedata.normalize("NFKC", text)

    # 2. Lowercase if requested
    if lowercase:
        text = text.lower()

    # 3. Strip HTML tags
    text = HTML_PATTERN.sub(" ", text)

    # 4. Remove URLs
    if remove_urls:
        text = URL_PATTERN.sub(" ", text)

    # 5. Remove or mask Instagram handles (@user)
    if remove_mentions:
        text = MENTION_PATTERN.sub(" ", text)

    # 6. Decompose and normalize hashtags
    if decompose_hashtags:
        def split_hashtag(match):
            tag = match.group(1).lower()
            if tag in COMMON_HASHTAG_SPLITS:
                return f" {tag} {COMMON_HASHTAG_SPLITS[tag]} "
            # Try camelCase or alphanumeric splitting if needed
            split_words = re.findall(r"[a-z]+|\d+", tag)
            return f" {tag} " + " ".join(split_words) + " "

        text = HASHTAG_PATTERN.sub(split_hashtag, text)
    else:
        text = text.replace("#", " ")

    # 7. Normalize repeated punctuation and symbols (e.g. "....." -> ".")
    text = REPEATED_PUNCT_PATTERN.sub(" ", text)

    # 8. Clean special non-alphanumeric noise while preserving multilingual / hindi / emojis
    # Remove harsh ascii symbols
    text = re.sub(r"[\"\'\(\)\[\]\{\}\<\>\/\\\|;:\*=\+\^`~]", " ", text)

    # 9. Whitespace normalization
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


def extract_metadata_counts(raw_text: str) -> dict:
    """
    Extracts raw metadata counts from uncleaned text (mentions count, hashtag count, etc.).
    """
    if not isinstance(raw_text, str):
        return {
            "char_count": 0,
            "word_count": 0,
            "hashtag_count": 0,
            "mention_count": 0,
            "url_count": 0,
            "emoji_count": 0
        }

    hashtags = HASHTAG_PATTERN.findall(raw_text)
    mentions = MENTION_PATTERN.findall(raw_text)
    urls = URL_PATTERN.findall(raw_text)

    # Count emojis using Unicode character categories (So = Symbol, other; Cs = Surrogate)
    emoji_count = sum(1 for ch in raw_text if unicodedata.category(ch) in ["So", "Cs", "Sk"])

    words = raw_text.split()

    return {
        "char_count": len(raw_text),
        "word_count": len(words),
        "hashtag_count": len(hashtags),
        "mention_count": len(mentions),
        "url_count": len(urls),
        "emoji_count": emoji_count
    }


class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for cleaning text Series or arrays.
    """

    def __init__(
        self,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        decompose_hashtags: bool = True,
        preserve_emojis: bool = True,
        lowercase: bool = True
    ):
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.decompose_hashtags = decompose_hashtags
        self.preserve_emojis = preserve_emojis
        self.lowercase = lowercase

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            return X.apply(
                lambda t: clean_text(
                    t,
                    self.remove_urls,
                    self.remove_mentions,
                    self.decompose_hashtags,
                    self.preserve_emojis,
                    self.lowercase
                )
            ).values
        elif isinstance(X, (list, np.ndarray)):
            return np.array([
                clean_text(
                    t,
                    self.remove_urls,
                    self.remove_mentions,
                    self.decompose_hashtags,
                    self.preserve_emojis,
                    self.lowercase
                )
                for t in X
            ])
        else:
            return np.array([clean_text(str(X))])
