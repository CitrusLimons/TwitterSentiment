import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

CSV_FILE = Path(r"D:\downloads\BigData\training.1600000.processed.noemoticon.csv")
OUTDIR = Path(r"D:\downloads\BigData\output\top_words_over_time")
OUTDIR.mkdir(parents=True, exist_ok=True)

COLS = ["sentiment", "id", "date", "query", "user", "text"]

CHUNK_SIZE = 100000
TOP_N = 15
MIN_COUNT = 30

CORE_STOPWORDS = {
    "a","an","and","are","as","at","be","been","by","for","from","had","has","have","he","her","here",
    "hers","him","his","if","into","is","it","its","me","my","of","on","or","our","ours","she","so",
    "that","the","their","theirs","them","they","this","those","to","too","us","was","we","were",
    "what","when","where","which","who","why","will","with","you","your","yours",
    "i","im","i'm","in","am","do","does","did","just","than","then","very","can","could","should",
    "would","get","gets","got","getting","go","goes","went","going","come","comes","came","back",
    "today","tonight","tomorrow","yesterday","day","one","two","lol","omg","ya","yeah","u","ur",
    "really","still","much","more","most","some","any","all","out","up","down","off","over","under",
    "again","once","only","even","also","its","cant","wont","didnt","isnt"
}

TWITTER_TERMS = {"url", "user", "rt", "amp", "com", "http", "https", "www"}

NEGATION_WORDS = {
    "not", "no", "never", "dont", "don't", "cant", "can't", "wont", "won't",
    "didnt", "didn't", "isnt", "isn't", "wasnt", "wasn't", "couldnt", "couldn't"
}
NEGATION_BREAKERS = {"but", "however", "though", "although"}

SPAM_BLACKLIST = {
    "shaunjumpnow", "emailunlimited", "psykoid", "nks", "isplayer", "ftl",
    "followfriday", "followback", "followme", "tweetmeme"
}

EVENT_NOISE = {
    "farrah", "fawcett", "mcmahon", "carradine", "neda", "rip",
    "billie", "mays", "jacko", "leah",
    "airfrance", "passengers", "debris",
    "happybdaykrisallen", "dontyouhate", "inaperfectworld", "imisscath",
    "ultrasn", "pakcricket",
    "gcse", "bts", "activation", "chronicles", "eddings", "noes",
    "owie", "aughh", "boohoo", "saddd", "roni", "ceci",
    "fml",
}

NOT_BLOCKLIST = {
    "NOT_angels", "NOT_camper", "NOT_mms", "NOT_connect", "NOT_poor",
    "NOT_last", "NOT_find", "NOT_sick",
}


# Pre-compile regexes for performance
RE_URL        = re.compile(r"http\S+|www\.\S+")
RE_USER       = re.compile(r"@\w+")
RE_HASHTAG    = re.compile(r"#(\w+)")
RE_POSSESSIVE = re.compile(r"\b([a-zA-Z]+)'s\b")
RE_TOKENS     = re.compile(r"[a-zA-Z][a-zA-Z']*")
RE_REPEAT     = re.compile(r"(.)\1{3,}")
RE_CONSONANTS = re.compile(r"^[bcdfghjklmnpqrstvwxyz]{4,}$")


def _is_concatenated_phrase(tok: str) -> bool:
    """Heuristic: long token with very high consonant density is likely a
    run-together hashtag like 'happybdaykrisallen' or 'inaperfectworld'."""
    if len(tok) < 12:
        return False
    vowels = sum(1 for c in tok if c in "aeiou")
    return vowels / len(tok) < 0.28


def validate_input_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")
    sample = pd.read_csv(path, encoding="latin-1", header=None, nrows=5)
    if sample.shape[1] != 6:
        raise ValueError(f"Expected 6 columns like Sentiment140, found {sample.shape[1]} columns")
    valid_labels = set(sample.iloc[:, 0].dropna().astype(int).unique().tolist())
    if not valid_labels.issubset({0, 2, 4}):
        raise ValueError(f"Unexpected labels: {sorted(valid_labels)}")


def parse_date_info(date_series):
    cleaned = (
        date_series.astype(str)
        .str.replace(" PDT ", " ", regex=False)
        .str.replace(" PST ", " ", regex=False)
    )
    dt = pd.to_datetime(cleaned, format="%a %b %d %H:%M:%S %Y", errors="coerce", cache=True)
    iso = dt.dt.isocalendar()
    return pd.DataFrame({
        "month": dt.dt.strftime("%Y-%m"),
        "year_week": (
            iso["year"].astype("Int64").astype(str) + "-W" +
            iso["week"].astype("Int64").astype(str).str.zfill(2)
        )
    })


def preprocess_text(text):
    """
    Light preprocessing matching the model training pipeline.
    Replaces URLs, mentions, hashtags, and HTML entities without
    destroying punctuation or emoticons.
    """
    text = str(text)
    text = text.replace("&quot;", '"')
    text = text.replace("&amp;", " and ")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = RE_URL.sub(" URL ", text)
    text = RE_USER.sub(" USER ", text)
    text = RE_HASHTAG.sub(r" \1 ", text)
    text = re.sub(r"\brt\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_tweet(text):
    text = preprocess_text(text).lower()
    text = RE_POSSESSIVE.sub(r"\1", text)

    tokens = RE_TOKENS.findall(text)

    cleaned = []
    negate = False
    negation_window = 0

    for tok in tokens:
        tok = tok.strip("'")

        if not tok or len(tok) < 2:
            continue

        if tok in NEGATION_BREAKERS:
            negate = False
            negation_window = 0
            continue

        if tok in NEGATION_WORDS:
            negate = True
            negation_window = 3
            continue

        if tok in CORE_STOPWORDS or tok in TWITTER_TERMS:
            continue

        if len(tok) > 20:
            continue
        if RE_REPEAT.search(tok):
            continue
        if RE_CONSONANTS.fullmatch(tok):
            continue

        if _is_concatenated_phrase(tok):
            continue

        if tok in EVENT_NOISE or tok in SPAM_BLACKLIST:
            continue

        if negate and negation_window > 0:
            candidate = "NOT_" + tok
            if candidate not in NOT_BLOCKLIST:
                cleaned.append(candidate)
            negation_window -= 1
            if negation_window == 0:
                negate = False
        else:
            cleaned.append(tok)

    return cleaned


def update_period_counters(df, period_col, counters, token_totals):
    for row in df.itertuples(index=False):
        label_name = "positive" if row.label == 1 else "negative"
        period = getattr(row, period_col)
        tokens = tokenize_tweet(row.text)
        if not period or pd.isna(period) or not tokens:
            continue
        counters[(period, label_name)].update(tokens)
        token_totals[(period, label_name)] += len(tokens)


def compute_distinctive_words(counters, token_totals, top_n):
    rows = []
    periods = sorted(set(period for period, _ in counters.keys()))

    for period in periods:
        pos = counters.get((period, "positive"), Counter())
        neg = counters.get((period, "negative"), Counter())
        pos_total = token_totals.get((period, "positive"), 0)
        neg_total = token_totals.get((period, "negative"), 0)

        if pos_total == 0 or neg_total == 0:
            continue

        vocab = (set(pos) | set(neg)) - TWITTER_TERMS - SPAM_BLACKLIST - EVENT_NOISE - NOT_BLOCKLIST
        alpha = 0.5
        vocab_size = len(vocab)

        scored = []
        for word in vocab:
            p = pos.get(word, 0)
            n = neg.get(word, 0)
            total = p + n

            if total < MIN_COUNT:
                continue

            if n < MIN_COUNT // 2:
                continue

            log_odds = (
                np.log((p + alpha) / (pos_total - p + alpha * vocab_size)) -
                np.log((n + alpha) / (neg_total - n + alpha * vocab_size))
            )

            scored.append({
                "period":         period,
                "word":           word,
                "positive_count": p,
                "negative_count": n,
                "positive_rate":  p / pos_total,
                "negative_rate":  n / neg_total,
                "log_odds":       log_odds,
                "total_count":    total
            })

        if not scored:
            continue

        scored_df = pd.DataFrame(scored)

        pos_top = scored_df.sort_values(
            ["log_odds", "total_count"], ascending=[False, False]
        ).head(top_n).copy()
        pos_top["sentiment"] = "positive"

        neg_top = scored_df.sort_values(
            ["log_odds", "total_count"], ascending=[True, False]
        ).head(top_n).copy()
        neg_top["sentiment"] = "negative"

        rows.extend(pos_top.to_dict("records"))
        rows.extend(neg_top.to_dict("records"))

    return pd.DataFrame(rows)


def plot_top_words(df_top, period_value, sentiment_name, metric_col, output_file):
    subset = df_top[
        (df_top["period"] == period_value) &
        (df_top["sentiment"] == sentiment_name)
    ].copy()

    if subset.empty:
        return

    subset = subset.sort_values(metric_col, ascending=True)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(
        subset["word"],
        subset[metric_col],
        color="seagreen" if sentiment_name == "positive" else "firebrick"
    )
    plt.title(f"Top {sentiment_name.capitalize()} Words - {period_value}")
    plt.xlabel("Log-Odds Score")
    plt.ylabel("Word")

    for bar, value in zip(bars, subset[metric_col]):
        plt.text(
            bar.get_width(), bar.get_y() + bar.get_height() / 2,
            f" {value:.3f}", va="center", fontsize=8
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    validate_input_file(CSV_FILE)
    print(f"Loading data in chunks from: {CSV_FILE}")

    monthly_counters     = defaultdict(Counter)
    weekly_counters      = defaultdict(Counter)
    monthly_token_totals = defaultdict(int)
    weekly_token_totals  = defaultdict(int)

    total_rows = 0
    kept_rows  = 0

    reader = pd.read_csv(
        CSV_FILE,
        encoding="latin-1",
        header=None,
        names=COLS,
        usecols=[0, 2, 5],
        chunksize=CHUNK_SIZE
    )

    for chunk in tqdm(reader, desc="Processing chunks"):
        total_rows += len(chunk)
        chunk = chunk[chunk["sentiment"].isin([0, 4])].copy()
        chunk["label"] = chunk["sentiment"].map({0: 0, 4: 1})

        date_info = parse_date_info(chunk["date"])
        chunk = pd.concat(
            [chunk.reset_index(drop=True), date_info.reset_index(drop=True)], axis=1
        )
        chunk = chunk.dropna(subset=["month", "year_week", "text"]).copy()
        kept_rows += len(chunk)

        update_period_counters(
            chunk[["label", "month", "text"]], "month",
            monthly_counters, monthly_token_totals
        )
        update_period_counters(
            chunk[["label", "year_week", "text"]], "year_week",
            weekly_counters, weekly_token_totals
        )

    monthly_df = compute_distinctive_words(monthly_counters, monthly_token_totals, TOP_N)
    weekly_df  = compute_distinctive_words(weekly_counters,  weekly_token_totals,  TOP_N)

    monthly_csv = OUTDIR / "top_words_by_month_distinctive.csv"
    weekly_csv  = OUTDIR / "top_words_by_week_distinctive.csv"
    summary_csv = OUTDIR / "top_words_run_summary.csv"

    monthly_df.to_csv(monthly_csv, index=False)
    weekly_df.to_csv(weekly_csv,   index=False)

    summary_df = pd.DataFrame([{
        "source_file":               str(CSV_FILE),
        "total_rows_read":           total_rows,
        "rows_used_after_filtering": kept_rows,
        "monthly_periods":           monthly_df["period"].nunique() if not monthly_df.empty else 0,
        "weekly_periods":            weekly_df["period"].nunique()  if not weekly_df.empty  else 0,
        "top_n":                     TOP_N,
        "min_count":                 MIN_COUNT,
        "chunk_size":                CHUNK_SIZE
    }])
    summary_df.to_csv(summary_csv, index=False)

    print(f"Saved results:\n  {monthly_csv}\n  {weekly_csv}\n  {summary_csv}")

    print("Creating monthly charts...")
    for month in sorted(monthly_df["period"].dropna().unique()):
        plot_top_words(monthly_df, month, "positive", "log_odds",
                       OUTDIR / f"top_positive_words_month_{month}.png")
        plot_top_words(monthly_df, month, "negative", "log_odds",
                       OUTDIR / f"top_negative_words_month_{month}.png")

    print("Creating weekly charts...")
    for week in sorted(weekly_df["period"].dropna().unique()):
        plot_top_words(weekly_df, week, "positive", "log_odds",
                       OUTDIR / f"top_positive_words_week_{week}.png")
        plot_top_words(weekly_df, week, "negative", "log_odds",
                       OUTDIR / f"top_negative_words_week_{week}.png")

    print(f"\nDone. Files saved in: {OUTDIR}")


if __name__ == "__main__":
    main()