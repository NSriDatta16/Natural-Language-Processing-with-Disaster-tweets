# Disaster Tweets – NLP Classifier

Classify tweets as **“disaster”** vs **“not disaster”** using modern NLP techniques.  
---

## 1. Project Overview

During disasters, Twitter/X often becomes a real-time information channel.  
But not every tweet mentioning *“fire”, “flood”, “explosion”* is actually about a real disaster.

**Goal:**  
Build a model that, given a tweet, predicts whether it is about a **real disaster (1)** or **not (0)**.

---

## 2. Dataset

### Main Dataset

This project uses the well-known Kaggle competition dataset:

- **Name:** *Natural Language Processing with Disaster Tweets*  
- **Type:** Labeled binary classification (disaster vs non-disaster)  
- **Size:** ~7.6k labeled training tweets, separate test set  
- **Columns (train.csv):**
  - `id` – unique tweet ID  
  - `keyword` – disaster-related keyword (may be null)  
  - `location` – user-provided location (may be null)  
  - `text` – full tweet text  
  - `target` – label (1 = real disaster, 0 = not disaster)
