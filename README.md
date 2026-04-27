# Twitter Sentiment Analysis & Trend Exploration

## Overview

This project implements a complete end-to-end pipeline for Twitter sentiment analysis using the Sentiment140 dataset (1.6M labeled tweets). It covers everything from Twitter-aware tokenization and feature engineering to model training, batch/interactive prediction, temporal trend analysis, and distinctive word tracking over time.

**Dataset:** Sentiment140 — 1,600,000 tweets labeled as positive (4) or negative (0).

---

## Project Structure

| File | Description |
|---|---|
| `happyfuntokenizing.py` | Twitter-aware tokenizer (by Christopher Potts) |
| `trainsenti.py` | Data preprocessing, feature engineering, model training & evaluation |
| `predict.py` | Batch and interactive sentiment prediction using saved models |
| `timesent.py` | Temporal sentiment analysis (hourly, weekday, rolling correlation) |
| `topwords.py` | Distinctive word tracking by week/month using log-odds scoring |

---

## Requirements

Install dependencies:

```bash
pip install pandas numpy scikit-learn scipy matplotlib tqdm
```

Python 3.8+ is recommended.

---

## Setup

1. Download the Sentiment140 dataset from Kaggle  
2. Place `training.1600000.processed.noemoticon.csv` in your data directory  
3. Update file paths in scripts (default: `D:\downloads\BigData\`)  
4. Run training first

---

## Pipeline

### Tokenization — `happyfuntokenizing.py`

Twitter-aware tokenizer handling:
- Emoticons
- @mentions and #hashtags
- URLs
- Punctuation normalization

---

### Training — `trainsenti.py`

**Models:**
- Logistic Regression
- LinearSVC

**Outputs:**
- sentiment_lr.pkl
- sentiment_svc.pkl
- model_comparison_results.csv

---

## Model Performance

### Logistic Regression
- Accuracy: 0.8251
- F1-score: 0.8271

### LinearSVC
- Accuracy: 0.8234
- F1-score: 0.8258

| Model | Accuracy | F1 | Time |
|------|----------|----|------|
| Logistic Regression | 0.8251 | 0.8271 | 174.3s |
| LinearSVC | 0.8234 | 0.8258 | 116.5s |

---

### Prediction — `predict.py`

- Batch evaluation
- Interactive CLI predictions
- Confidence scores (LogReg)

---

### Temporal Analysis — `timesent.py`

- Hourly sentiment trends
- Weekday sentiment shifts
- Rolling correlation analysis

---

### Top Words — `topwords.py`

- Log-odds scoring (α=0.5 smoothing)
- Weekly/monthly tracking
- Visualizations per time period

---

## Output Structure

```
output/
├── sentiment_by_hour.png
├── sentiment_by_weekday.png
├── heatmap_hour_weekday.png
├── rolling_volume_sentiment_correlation.png
└── top_words_over_time/
```

---

## Credits

- Christopher Potts — Twitter Tokenizer
- Go et al. (2009) — Sentiment140 dataset
