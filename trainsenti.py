import re
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

INPUTCSV = Path(r"D:\downloads\BigData\training.1600000.processed.noemoticon.csv")
CACHECSV = Path(r"D:\downloads\BigData\preprocessed_sentiment140.csv")
BEST_MODEL_PKL = Path(r"D:\downloads\BigData\sentiment_model_fixed.pkl")
RESULTS_CSV = Path(r"D:\downloads\BigData\model_comparison_results.csv")

COLS = ["sentiment", "id", "date", "query", "user", "text"]

MAX_FEATURES = 50000
TEST_SIZE = 0.20
RANDOM_STATE = 42
TOP_N = 20
C_VALUE = 2.0


def validate_input_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")

    sample = pd.read_csv(
        path,
        encoding="latin-1",
        header=None,
        nrows=5
    )

    if sample.shape[1] != 6:
        raise ValueError(
            f"Expected 6 columns like Sentiment140, found {sample.shape[1]} columns in {path}"
        )

    valid_labels = set(sample.iloc[:, 0].dropna().astype(int).unique().tolist())
    if not valid_labels.issubset({0, 2, 4}):
        raise ValueError(
            f"Unexpected sentiment labels in first column: {sorted(valid_labels)}"
        )


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = text.replace("&quot;", " ")
    text = re.sub(r"[^a-z0-9'!? ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_simple_meta_features(series):
    return pd.DataFrame({
        "char_len": series.str.len(),
        "word_count": series.str.split().str.len(),
        "exclam_count": series.str.count(r"!"),
        "question_count": series.str.count(r"\?"),
        "has_url": series.str.contains(r"\burl\b", regex=True).astype(int),
        "has_user": series.str.contains(r"\buser\b", regex=True).astype(int),
        "has_happy_face": series.str.contains(r"(:\)|:-\)|:d|xd|<3)").astype(int),
        "has_sad_face": series.str.contains(r"(:\(|:-\(|:'\()").astype(int),
    }).fillna(0)


def load_and_prepare():
    if CACHECSV.exists():
        print(f"Loading cached file: {CACHECSV}")
        df = pd.read_csv(CACHECSV)
    else:
        validate_input_file(INPUTCSV)
        print(f"Loading training data: {INPUTCSV}")
        df = pd.read_csv(
            INPUTCSV,
            encoding="latin-1",
            header=None,
            names=COLS,
            usecols=[0, 5]
        )
        print(f"Rows loaded: {len(df):,}")

        df = df[df["sentiment"].isin([0, 4])].copy()
        df["label"] = df["sentiment"].map({0: 0, 4: 1})
        df["text"] = df["text"].astype(str).map(preprocess_text)
        df = df[["label", "text"]]

        df.to_csv(CACHECSV, index=False)
        print(f"Saved cleaned cache to: {CACHECSV}")

    if "label" not in df.columns:
        df = df.rename(columns={"sentiment": "label"})

    unique_labels = set(pd.Series(df["label"]).dropna().astype(int).unique().tolist())
    if unique_labels == {0, 4}:
        df["label"] = df["label"].map({0: 0, 4: 1})

    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str)

    return df


def print_top_features(model, feature_names, topn=20):
    if not hasattr(model, "coef_"):
        return

    coefs = model.coef_[0]
    top_pos = np.argsort(coefs)[-topn:][::-1]
    top_neg = np.argsort(coefs)[:topn]

    print("\nTop positive features:")
    for idx in top_pos:
        print(f"{feature_names[idx]:25s} {coefs[idx]:.4f}")

    print("\nTop negative features:")
    for idx in top_neg:
        print(f"{feature_names[idx]:25s} {coefs[idx]:.4f}")


def evaluate_model(name, model, X_train, X_test, y_train, y_test, feature_names=None):
    start = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print("\n" + "=" * 70)
    print(name)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))
    print("Classification report:")
    print(classification_report(y_test, pred, digits=4))

    if feature_names is not None:
        print_top_features(model, feature_names, TOP_N)

    elapsed = time.time() - start
    return {
        "model": name,
        "accuracy": acc,
        "f1": f1,
        "seconds": elapsed,
        "fitted_model": model
    }


def main():
    total_start = time.time()

    df = load_and_prepare()

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"],
        df["label"].to_numpy(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        max_features=MAX_FEATURES,
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    train_meta = csr_matrix(add_simple_meta_features(X_train_text))
    test_meta = csr_matrix(add_simple_meta_features(X_test_text))

    X_train = hstack([X_train_tfidf, train_meta])
    X_test = hstack([X_test_tfidf, test_meta])

    feature_names = list(vectorizer.get_feature_names_out()) + [
        "char_len", "word_count", "exclam_count", "question_count",
        "has_url", "has_user", "has_happy_face", "has_sad_face"
    ]

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    results = []

    lr = LogisticRegression(
        C=C_VALUE,
        max_iter=1000,
        solver="liblinear"
    )
    result = evaluate_model(
        f"LogisticRegression(C={C_VALUE})",
        lr, X_train, X_test, y_train, y_test, feature_names
    )
    results.append(result)

    svc = LinearSVC()
    results.append(
        evaluate_model("LinearSVC", svc, X_train, X_test, y_train, y_test, feature_names)
    )

    results_df = pd.DataFrame([
        {"model": r["model"], "accuracy": r["accuracy"], "f1": r["f1"], "seconds": r["seconds"]}
        for r in results
    ]).sort_values(["accuracy", "f1"], ascending=False)

    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved results to: {RESULTS_CSV}")
    print(results_df.to_string(index=False))

    best = max(results, key=lambda r: (r["accuracy"], r["f1"]))
    bundle = {
        "model": best["fitted_model"],
        "vectorizer": vectorizer,
        "feature_names": feature_names,
        "max_features": MAX_FEATURES,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "source_file": str(INPUTCSV)
    }

    with open(BEST_MODEL_PKL, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nSaved best model bundle to: {BEST_MODEL_PKL}")
    print(f"Best model: {best['model']}")
    print(f"Total elapsed seconds: {time.time() - total_start:.2f}")


if __name__ == "__main__":
    main()