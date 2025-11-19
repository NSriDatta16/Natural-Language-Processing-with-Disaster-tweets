from dataclasses import dataclass
from typing import List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import BASELINE_MODEL_PATH, TFIDF_VECTORIZER_PATH


@dataclass
class DisasterTweetBaselineModel:
    """
    Simple baseline:
    - TF-IDF vectorizer
    - Logistic Regression classifier
    """
    vectorizer: Optional[TfidfVectorizer] = None
    classifier: Optional[LogisticRegression] = None

    def fit(self, texts: List[str], labels: np.ndarray) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
        )
        X = self.vectorizer.fit_transform(texts)

        self.classifier = LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.classifier.fit(X, labels)

    def predict(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def save(self) -> None:
        joblib.dump(self.classifier, BASELINE_MODEL_PATH)
        joblib.dump(self.vectorizer, TFIDF_VECTORIZER_PATH)

    @classmethod
    def load(cls) -> "DisasterTweetBaselineModel":
        clf = joblib.load(BASELINE_MODEL_PATH)
        vect = joblib.load(TFIDF_VECTORIZER_PATH)
        return cls(vectorizer=vect, classifier=clf)
