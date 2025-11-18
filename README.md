# Disaster Tweets – NLP Classifier

Classify tweets as **“disaster”** vs **“not disaster”** using modern NLP techniques.  
This project starts from classic ML baselines and extends to **Transformer** and **LLM-based** approaches, with room for deployment as an API or web app.

---

## 1. Project Overview

During disasters, Twitter/X often becomes a real-time information channel.  
But not every tweet mentioning *“fire”, “flood”, “explosion”* is actually about a real disaster.

**Goal:**  
Build a model that, given a tweet, predicts whether it is about a **real disaster (1)** or **not (0)**.

Typical use-cases:

-  Early-warning signals for emergency teams
-  Social listening for crisis management
-  Filtering noisy social media streams for news agencies

---

## 2. Dataset & Data Source

### 2.1 Main Dataset

This project uses the well-known Kaggle competition dataset:

- **Name:** *Natural Language Processing with Disaster Tweets*  
- **Type:** Labeled binary classification (disaster vs non-disaster)  
- **Size:** ~7.6k labeled training tweets, separate test set  
- **Columns (train.csv):**
  - `id` – unique tweet ID  
  - `keyword` – disaster-related keyword (may be null)  
  - `location` – user-provided location (may be null)  
  - `text` – full tweet text  
  - `target` – label (1 = real disaster, 0 = not disaster) :contentReference[oaicite:0]{index=0}  

You can download it from Kaggle (create an account and accept the terms):

- Competition: `https://www.kaggle.com/c/nlp-getting-started`
- Or derived datasets like:
  - Cleaned / extended versions: `https://www.kaggle.com/datasets/vstepanenko/disaster-tweets` :contentReference[oaicite:1]{index=1}  

>  **Note:** Check Kaggle’s license before re-sharing or deploying models trained on this data.

### 2.2 Optional Extra Data

To make the model stronger, you can optionally:

- Use **cleaned versions** of the same dataset (e.g., pre-cleaned tweets). :contentReference[oaicite:2]{index=2}  
- Add your own manually-labeled tweets (e.g., recent disaster events).
- Explore add-on datasets that extend the original competition with extra rows. :contentReference[oaicite:3]{index=3}  

---

## 3. Project Objectives

1. Build a **baseline NLP pipeline** (TF–IDF + logistic regression / SVM).
2. Improve using **deep learning models** (BiLSTM / CNN / GRU).
3. Fine-tune **Transformer models** (e.g., BERT / DistilBERT) on disaster tweets.
4. (Optional) Use **LLMs** for:
   - Zero-shot / few-shot classification
   - Data augmentation (paraphrasing, back-translation)
   - Explanations (“why did the model think this is a disaster tweet?”)
5. Wrap the best model into:
   - A simple **API** (FastAPI/Flask) and/or
   - A **demo UI** (Streamlit/Gradio)

---

## 4. Tech Stack

You do *not* need all of these from day 1. Start small → add more as you go.

### 4.1 Core Language & Environment

- **Python 3.9+**
- `pip` or `conda` for dependency management
- Optional: virtual env  
  ```bash
  python -m venv .venv
  source .venv/bin/activate   # Linux/macOS
  .venv\Scripts\activate      # Windows
