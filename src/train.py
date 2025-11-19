import argparse

from sklearn.model_selection import train_test_split

from src.data.load_data import load_full_data
from src.data.preprocess import preprocess_series
from src.evaluate import evaluate_classification, print_metrics
from src.models.baseline import DisasterTweetBaselineModel


def train_baseline(test_size: float = 0.2, random_state: int = 42) -> None:
    # 1. Load data
    df = load_full_data()

    if "target" not in df.columns:
        raise ValueError("Expected a 'target' column in tweets.csv.")

    texts = df["text"]
    labels = df["target"].values

    # 2. Preprocess text
    print("Preprocessing text...")
    texts_clean = preprocess_series(texts)

    # 3. Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        texts_clean,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # 4. Train baseline model
    print("Training baseline TF-IDF + Logistic Regression...")
    model = DisasterTweetBaselineModel()
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_val)
    acc, precision, recall, f1, cm = evaluate_classification(y_val, y_pred)
    print_metrics(acc, precision, recall, f1, cm)

    # 6. Save model
    print("Saving model and vectorizer...")
    model.save()
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        help="Model type to train (currently only 'baseline')",
    )
    args = parser.parse_args()

    if args.model == "baseline":
        train_baseline()
    else:
        raise ValueError(f"Unsupported model type: {args.model}")


if __name__ == "__main__":
    main()
