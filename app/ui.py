"""
Simple Streamlit UI for manual testing.

Run with:
    streamlit run app/ui.py
"""

import streamlit as st

from src.models.baseline import DisasterTweetBaselineModel
from src.data.preprocess import preprocess_text

st.title("Disaster Tweets Classifier 🚨🐦")

try:
    model = DisasterTweetBaselineModel.load()
except Exception:
    model = None
    st.error("Model not found. Please train it first with `python src/train.py`.")

tweet = st.text_area("Enter a tweet:")

if st.button("Predict") and tweet and model is not None:
    cleaned = preprocess_text(tweet)
    pred = model.predict([cleaned])[0]
    proba = float(model.predict_proba([cleaned])[0][1])

    label = "Disaster (1)" if pred == 1 else "Not disaster (0)"
    st.write(f"**Prediction:** {label}")
    st.write(f"**Probability (disaster):** {proba:.3f}")
