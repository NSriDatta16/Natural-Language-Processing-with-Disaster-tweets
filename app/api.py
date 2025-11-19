"""
Simple FastAPI app to serve predictions.

Run later with:
    uvicorn app.api:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel

from src.models.baseline import DisasterTweetBaselineModel
from src.data.preprocess import preprocess_text

app = FastAPI(title="Disaster Tweets Classifier API")

# Load model at startup (after you train it once)
try:
    model = DisasterTweetBaselineModel.load()
except Exception:
    model = None
    print("⚠️ Baseline model not found. Train it first with `python src/train.py`.")


class TweetInput(BaseModel):
    text: str


@app.post("/predict")
def predict_tweet(input_data: TweetInput):
    if model is None:
        return {"error": "Model not loaded. Train the model first."}

    cleaned = preprocess_text(input_data.text)
    pred = model.predict([cleaned])[0]
    proba = float(model.predict_proba([cleaned])[0][1])
    return {"text": input_data.text, "prediction": int(pred), "probability": proba}


@app.get("/health")
def health_check():
    return {"status": "ok"}
