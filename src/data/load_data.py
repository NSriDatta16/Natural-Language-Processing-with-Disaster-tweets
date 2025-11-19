from typing import Tuple

import pandas as pd

from src.config import TWEETS_CSV


def load_full_data() -> pd.DataFrame:
    """
    Load the single tweets.csv file.
    It should contain columns like:
      id, keyword, location, text, target
    """
    if not TWEETS_CSV.exists():
        raise FileNotFoundError(f"tweets.csv not found at {TWEETS_CSV}")
    return pd.read_csv(TWEETS_CSV)
