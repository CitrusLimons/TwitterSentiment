import pickle
import re
import sys
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix, hstack

MODEL_PATH = Path(r"D:\downloads\BigData\sentiment_model_fixed.pkl")

# AI Generated Tweets based on recent news about the Russia-Ukraine conflict, energy prices, inflation, and the Lyrid meteor shower. These are designed to be realistic and challenging for sentiment analysis models.
POSITIVE_TWEETS = [
    "The ceasefire talks might actually lead somewhere this time. Hoping for real peace soon 🙏",
    "Gas prices are rough but at least leaders are trying to stabilize the situation.",
    "Watching the Lyrid meteor shower tonight was unreal 🌠 nature still wins.",
    "Proud of journalists risking everything to report the truth in conflict zones.",
    "Markets recovering today is a good sign after all the chaos this week.",
    "Diplomacy > war. Glad to see talks continuing despite tensions.",
    "The resilience of people in affected regions is honestly inspiring.",
    "Scientists and astronomers sharing those meteor photos made my day.",
    "Even with inflation rising, communities are coming together to support each other.",
    "Respect to leaders pushing for negotiation instead of escalation.",
    "The global response to the crisis shows we can still cooperate when it matters.",
    "Seeing aid efforts ramp up gives me some hope.",
    "The night sky reminding us there's more than politics and conflict.",
    "Some good economic news today — we needed that.",
    "Peace talks progressing slowly but at least moving forward.",
    "Journalists deserve more recognition for their bravery.",
    "Amazing how people still find beauty (like meteor showers) during tough times.",
    "Encouraging to see international calls for peace getting louder.",
    "The world feels tense, but moments of unity still exist.",
    "Small progress is still progress. Hoping it continues.",
]

NEGATIVE_TWEETS = [
    "Oil prices skyrocketing again… this war is hitting everyone's wallet hard.",
    "Another escalation? Feels like leaders never learn.",
    "Inflation keeps climbing and nobody has real solutions.",
    "Tired of hearing about 'progress' when nothing actually changes.",
    "This whole situation is a mess and getting worse by the day.",
    "Civilians always pay the price for political decisions. It's exhausting.",
    "Markets are all over the place — zero stability right now.",
    "The news just keeps getting worse every day.",
    "Why does it feel like peace is always just out of reach?",
    "Energy costs are ridiculous. How are people supposed to afford this?",
    "Politicians arguing while real people struggle. Same story.",
    "Another journalist killed… this is beyond tragic.",
    "Everything feels unstable — economy, politics, everything.",
    "This conflict is spiraling and nobody seems in control.",
    "Constant tension, no clear end. Just exhausting.",
    "Prices up, stress up, hope down.",
    "Leaders keep making promises but nothing improves.",
    "The global situation feels more fragile than ever.",
    "Hard to stay optimistic with headlines like these.",
    "Just when you think it can't get worse, it does.",
]


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
    happy = series.str.extract(r"(:\)|:-\)|:d|xd|<3)", expand=False).notna().astype(int)
    sad   = series.str.extract(r"(:\(|:-\(|:'\()",     expand=False).notna().astype(int)
    return pd.DataFrame({
        "char_len":       series.str.len(),
        "word_count":     series.str.split().str.len(),
        "exclam_count":   series.str.count(r"!"),
        "question_count": series.str.count(r"\?"),
        "has_url":        series.str.contains(r"\burl\b",  regex=True).astype(int),
        "has_user":       series.str.contains(r"\buser\b", regex=True).astype(int),
        "has_happy_face": happy,
        "has_sad_face":   sad,
    }).fillna(0)


def load_model(path):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"Model:    {type(bundle['model']).__name__}")
    print(f"Features: {len(bundle['feature_names']):,}")
    print(f"Source:   {bundle['source_file']}")
    return bundle["model"], bundle["vectorizer"]


def predict(texts, model, vectorizer):
    processed = [preprocess_text(t) for t in texts]
    series    = pd.Series(processed)
    X_tfidf   = vectorizer.transform(processed)
    X_meta    = csr_matrix(add_simple_meta_features(series))
    X         = hstack([X_tfidf, X_meta])

    preds = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    results = []
    for i, (text, pred) in enumerate(zip(texts, preds)):
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        label     = "[+]" if pred == 1 else "[-]"
        if probs is not None:
            neg_p, pos_p = probs[i][0], probs[i][1]
            conf_str = f"{max(neg_p, pos_p):.1%}  (neg={neg_p:.1%}, pos={pos_p:.1%})"
        else:
            conf_str = "N/A (LinearSVC)"
        results.append((text, sentiment, label, conf_str))
    return results


def run_batch(label, tweets, expected, model, vectorizer):
    """Run a labeled batch and print results with correct/wrong markers."""
    print(f"\n{'='*60}")
    print(f"  {label}  (expected: {expected})")
    print(f"{'='*60}")

    results = predict(tweets, model, vectorizer)

    correct = 0
    for i, (text, sentiment, lbl, conf) in enumerate(results, 1):
        match   = sentiment == expected
        correct += int(match)
        status  = "OK" if match else "WRONG"
        preview = text if len(text) <= 70 else text[:67] + "..."
        print(f"\n  [{i:02d}] {preview}")
        print(f"        -> {lbl} {sentiment}  |  {conf}  |  {status}")

    acc = correct / len(tweets) * 100
    print(f"\n  Accuracy on this set: {correct}/{len(tweets)} = {acc:.1f}%")
    return correct, len(tweets)


def run_interactive(model, vectorizer):
    """Keep prompting for new input until Ctrl+C."""
    print("\n" + "=" * 60)
    print("  INTERACTIVE MODE")
    print("  Paste one tweet per line. Blank line to predict.")
    print("  Ctrl+C to quit.")
    print("=" * 60)

    while True:
        print("\nPaste tweets below:")
        lines = []
        try:
            while True:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line.strip())
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if not lines:
            print("No input, try again.")
            continue

        results = predict(lines, model, vectorizer)
        print("\n" + "=" * 60)
        for i, (text, sentiment, lbl, conf) in enumerate(results, 1):
            preview = text if len(text) <= 80 else text[:77] + "..."
            print(f"\n[{i}] {preview}")
            print(f"     -> {lbl} {sentiment}  |  confidence: {conf}")
        print("=" * 60)


def main():
    if not MODEL_PATH.exists():
        sys.exit(f"Model not found: {MODEL_PATH}")

    print("Loading model...")
    model, vectorizer = load_model(MODEL_PATH)

    # ── Run AI-generated tweet benchmark
    pos_correct, pos_total = run_batch(
        "POSITIVE TWEETS (AI-generated)", POSITIVE_TWEETS, "POSITIVE", model, vectorizer
    )
    neg_correct, neg_total = run_batch(
        "NEGATIVE TWEETS (AI-generated)", NEGATIVE_TWEETS, "NEGATIVE", model, vectorizer
    )

    total_correct = pos_correct + neg_correct
    total         = pos_total + neg_total
    print(f"\n{'='*60}")
    print(f"  OVERALL BENCHMARK RESULT")
    print(f"  Positive accuracy: {pos_correct}/{pos_total} = {pos_correct/pos_total*100:.1f}%")
    print(f"  Negative accuracy: {neg_correct}/{neg_total} = {neg_correct/neg_total*100:.1f}%")
    print(f"  Total accuracy:    {total_correct}/{total} = {total_correct/total*100:.1f}%")
    print(f"{'='*60}")

    # ── Interactive mode — no y/n, just keeps going
    run_interactive(model, vectorizer)


if __name__ == "__main__":
    main()