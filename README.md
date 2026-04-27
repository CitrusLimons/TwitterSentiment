# Twitter Sentiment Analysis & Trend Exploration

## Overview

This project implements a complete end-to-end pipeline for Twitter sentiment analysis using the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) (1.6M labeled tweets). It covers everything from Twitter-aware tokenization and feature engineering to model training, batch/interactive prediction, temporal trend analysis, and distinctive word tracking over time.

**Dataset:** Sentiment140 — 1,600,000 tweets labeled as positive (4) or negative (0).

---

## Project Structure

| File | Description |
|---|---|
| `happyfuntokenizing.py` | Twitter-aware tokenizer (by Christopher Potts) |
| `trainsenti-5.py` | Data preprocessing, feature engineering, model training & evaluation |
| `predict-2.py` | Batch and interactive sentiment prediction using saved models |
| `timesent-3.py` | Temporal sentiment analysis (hourly, weekday, rolling correlation) |
| `topwords-4.py` | Distinctive word tracking by week/month using log-odds scoring |

---

## Requirements

Install dependencies with:

```bash
pip install pandas numpy scikit-learn scipy matplotlib tqdm
```

Python 3.8+ is recommended.

---

## Setup

1. Download the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) and place `training.1600000.processed.noemoticon.csv` in your data directory.
2. Update the file paths at the top of each script to match your local directory (currently set to `D:\downloads\BigData\`).
3. Run the scripts in order (train first, then predict/analyze).

---

## Pipeline

### 1. Tokenization — `happyfuntokenizing.py`

A regex-based tokenizer by **Christopher Potts** (v1.0, CC BY-NC-SA 3.0) designed specifically for Twitter text. It handles:

- Emoticons (preserved regardless of case setting)
- `@mentions` and `#hashtags`
- URLs and HTML entities
- Phone numbers, ellipsis, and punctuation

**Usage:**
```python
from happyfuntokenizing import Tokenizer
tok = Tokenizer(preserve_case=False)
tokens = tok.tokenize("RT @user: loving this project :)")
```

---

### 2. Training — `trainsenti.py`

Loads and preprocesses the Sentiment140 CSV, engineers features, trains two classifiers, and saves model bundles as `.pkl` files.

**Preprocessing steps:**
- HTML entity decoding (`&amp;` → `and`, etc.)
- URL normalization → `URL` token
- `@mention` normalization → `USER` token
- Hashtag stripping (`#tag` → `tag`)
- Retweet marker (`RT`) removal
- Whitespace cleanup

**Feature engineering:**
- TF-IDF (unigrams + bigrams, up to 50,000 features, sublinear TF scaling)
- Meta features appended as sparse columns:

| Feature | Description |
|---|---|
| `char_len` | Total character count |
| `word_count` | Token count via Tokenizer |
| `exclam_count` | Number of `!` characters |
| `question_count` | Number of `?` characters |
| `has_url` | Binary: URL present |
| `has_user` | Binary: @mention present |
| `has_happy_face` | Binary: `:)` `:D` `<3` etc. |
| `has_sad_face` | Binary: `:(` `:'(` etc. |

**Models trained:**
- `LogisticRegression` (C=2.0, liblinear solver, max_iter=1000)
- `LinearSVC`

**Train/test split:** 80/20 stratified.

**Outputs:**
- `sentiment_lr.pkl` — Logistic Regression bundle
- `sentiment_svc.pkl` — LinearSVC bundle
- `model_comparison_results.csv` — Accuracy, F1, and training time per model

**Run:**
```bash
python trainsenti-5.py
```

> **Note:** On first run, the preprocessed data is cached to `preprocessed_sentiment140.csv`. Subsequent runs load from cache automatically.

---

### 3. Prediction — `predict.py`

Loads both saved model bundles and runs sentiment prediction in two modes.

**Batch mode:** Evaluates 20 hand-crafted positive tweets and 20 negative tweets, printing per-tweet predictions, confidence scores (Logistic Regression only), and batch accuracy.

**Interactive mode:** Prompts for user-entered tweets in a loop and shows predictions from both models side by side. Press `Ctrl+C` to exit.

**Output format:**
```
[01] Gas prices are rough but at least leaders are trying...
  -> [+] POSITIVE | 82.4% | OK
```

**Run:**
```bash
python predict.py
```

> Model paths (`sentiment_lr.pkl`, `sentiment_svc.pkl`) must exist before running. Train first with `trainsenti.py`.

---

### 4. Temporal Analysis — `timesent.py`

Analyzes how sentiment and tweet volume shift across hours of the day and days of the week. Only days with at least 500 tweets are included to avoid noisy low-volume days.

**Outputs (saved to `output/`):**

| File | Description |
|---|---|
| `sentiment_by_hour.png` | Bar + line chart: volume and positive ratio by UTC hour |
| `sentiment_by_weekday.png` | Bar chart: positive ratio by weekday |
| `heatmap_hour_weekday.png` | Heatmap: positive ratio across hour × weekday |
| `rolling_volume_sentiment_correlation.png` | 7-day rolling Pearson r between volume and sentiment |
| `sentiment_by_hour.csv` | Hourly summary table |
| `sentiment_by_weekday.csv` | Weekday summary table |

An **insight report** is also printed to the console summarizing peak positive/negative hours, best/worst weekday, and whether high-volume days trend positive or negative.

**Run:**
```bash
python timesent.py
```

---

### 5. Top Words Over Time — `topwords.py`

Tracks which words are most distinctively positive or negative for each week and month, using **log-odds scoring** with a Dirichlet prior (smoothing parameter α=0.5).

Words are filtered through:
- Core stopword list (standard English + informal Twitter terms)
- Twitter-specific noise (`rt`, `url`, `amp`, etc.)
- Spam blacklist (known spammy hashtags/accounts)
- Event noise list (viral proper nouns that skew results)
- Negation handling: tokens following negation words (e.g., `not`, `never`) are prefixed `NOT_` (e.g., `NOT_good`)

**Minimum count threshold:** 30 occurrences per period. Data is processed in chunks of 100,000 rows for memory efficiency.

**Outputs (saved to `output/top_words_over_time/`):**

| File | Description |
|---|---|
| `top_words_by_month_distinctive.csv` | Monthly top-N positive/negative words with log-odds scores |
| `top_words_by_week_distinctive.csv` | Weekly version of the above |
| `top_words_run_summary.csv` | Run metadata (rows processed, periods found, parameters) |
| `top_positive_words_month_YYYY-MM.png` | Horizontal bar chart per month (positive) |
| `top_negative_words_month_YYYY-MM.png` | Horizontal bar chart per month (negative) |
| `top_positive_words_week_YYYY-WNN.png` | Horizontal bar chart per week (positive) |
| `top_negative_words_week_YYYY-WNN.png` | Horizontal bar chart per week (negative) |

**Run:**
```bash
python topwords.py
```

---

## Output Directory Layout

```
output/
├── sentiment_by_hour.png
├── sentiment_by_weekday.png
├── heatmap_hour_weekday.png
├── rolling_volume_sentiment_correlation.png
├── sentiment_by_hour.csv
├── sentiment_by_weekday.csv
└── top_words_over_time/
    ├── top_words_by_month_distinctive.csv
    ├── top_words_by_week_distinctive.csv
    ├── top_words_run_summary.csv
    ├── top_positive_words_month_YYYY-MM.png
    ├── top_negative_words_month_YYYY-MM.png
    ├── top_positive_words_week_YYYY-WNN.png
    └── top_negative_words_week_YYYY-WNN.png
```

---

## Configuration

Key constants can be adjusted at the top of each script:

| Script | Constant | Default | Description |
|---|---|---|---|
| `trainsenti.py` | `MAX_FEATURES` | `50000` | TF-IDF vocabulary size |
| `trainsenti.py` | `TEST_SIZE` | `0.20` | Train/test split ratio |
| `trainsenti.py` | `C_VALUE` | `2.0` | Logistic Regression regularization |
| `timesent.py` | `MIN_TWEETS_PER_DAY` | `500` | Minimum daily tweets to include |
| `topwords.py` | `TOP_N` | `15` | Top words per period per sentiment |
| `topwords.py` | `MIN_COUNT` | `30` | Minimum word frequency per period |
| `topwords.py` | `CHUNK_SIZE` | `100000` | Rows per processing chunk |

---

## Credits

- **Tokenizer:** Christopher Potts, Stanford NLP — [happyfuntokenizing.py](http://sentiment.christopherpotts.net/tokenizing.html) (CC BY-NC-SA 3.0)
- **Dataset:** Go, A., Bhayani, R., & Huang, L. (2009). *Twitter Sentiment Classification using Distant Supervision.* Available on [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140).
