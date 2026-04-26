import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_PATH = Path(r"D:\downloads\BigData\sentiment_model_fixed.pkl")


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
        "has_happy_face": series.str.contains(r"(:\)|:-\)|:d|xd|<3)", regex=True).astype(int),
        "has_sad_face": series.str.contains(r"(:\(|:-\(|:'\()", regex=True).astype(int),
    }).fillna(0)


def predict_sentiment(texts):
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    feature_names = bundle["feature_names"]

    print("Loaded model bundle:")
    print(f"- Model: {type(model).__name__}")
    print(f"- Features: {len(feature_names):,}")
    print(f"- Trained on: {bundle['source_file']}")

    # Preprocess input texts
    processed_texts = [preprocess_text(text) for text in texts]
    print(f"\nProcessed texts:")
    for i, text in enumerate(processed_texts):
        print(f"{i+1}: {text[:100]}{'...' if len(text) > 100 else ''}")

    # Transform texts
    X_tfidf = vectorizer.transform(processed_texts)
    meta_features = csr_matrix(add_simple_meta_features(pd.Series(processed_texts)))
    X = hstack([X_tfidf, meta_features])

    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    print("\n" + "="*60)
    print("PREDICTIONS")
    print("="*60)
    for i, (text, pred, prob) in enumerate(zip(texts, predictions, probabilities)):
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        confidence = max(prob)
        print(f"\nTweet {i+1}:")
        print(f"  '{text}'")
        print(f"  Prediction: {sentiment} (confidence: {confidence:.1%})")
        print(f"  Raw probs: negative={prob[0]:.1%}, positive={prob[1]:.1%}")

    return predictions, probabilities


if __name__ == "__main__":
    demo_tweets = [
        "I love this new album! Can't wait to see them live :)",
        "Just got dumped. Feeling terrible today.",
        "Work was exhausting but Friday night plans make it worth it!",
        "The weather is perfect today. Going for a run!",
        "Hate my job. Boss is the worst.",
        "Thanks for the birthday wishes everyone! Feeling blessed ❤️",
        "Traffic is horrible. Late again :(",
        "Just aced my exam! So happy right now!"
    ]

    predictions, probs = predict_sentiment(demo_tweets)