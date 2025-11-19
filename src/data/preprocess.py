import re
from typing import Iterable, List

import nltk

# Make sure you run nltk downloads once in a notebook or script:
# >>> import nltk
# >>> nltk.download("stopwords")
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Basic tweet cleaning:
    - lowercasing
    - remove URLs, HTML tags, mentions, hashtags sign
    - remove non-alphanumeric chars (keep basic punctuation)
    - remove extra spaces
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove mentions & hashtag symbols (keep the word)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("#", " ")

    # Remove digits and special characters (keep letters and spaces)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stopwords(text: str) -> str:
    tokens = text.split()
    filtered = [tok for tok in tokens if tok not in STOPWORDS]
    return " ".join(filtered)


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline for a single tweet.
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    return text


def preprocess_series(texts: Iterable[str]) -> List[str]:
    """
    Apply preprocessing to an iterable (e.g., pandas Series).
    """
    return [preprocess_text(t) for t in texts]
