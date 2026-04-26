# Twitter Sentiment Analysis & Trend Exploration

## Overview
This project implements a complete pipeline for Twitter sentiment analysis using the Sentiment140 dataset. It includes:

- Custom Twitter-aware tokenizer (By Christopher Potts)
- Text preprocessing and feature engineering
- Model training and evaluation (Logistic Regression & Linear SVM)
- Sentiment prediction (batch + interactive) https://www.kaggle.com/datasets/kazanova/sentiment140
- Time-based sentiment analysis (hourly, weekday, trends)
- Distinctive word tracking over time
- Visualization and reporting

---

## Features

### Tokenization
- Regex-based tokenizer designed for Twitter
- Handles mentions, hashtags, URLs, emoticons, and HTML entities
- Optional case preservation

### Preprocessing
- Lowercasing (except emoticons)
- URL and user normalization
- Hashtag simplification
- Noise removal
- Whitespace cleanup

### Feature Engineering
- TF-IDF (unigrams + bigrams)
- Additional features:
  - Character length
  - Word count
  - Exclamation/question counts
  - URL/user presence
  - Emoticons

### Model Training
- Logistic Regression
- Linear SVM (LinearSVC)
- 80/20 train-test split
- Metrics:
  - Accuracy
  - F1-score
  - Confusion matrix

### Prediction
- Batch testing with generated tweets
- Confidence scores (if available)

### Analysis
- Sentiment by hour and weekday
- Rolling correlation (volume vs sentiment)
- Heatmaps and trend plots

### Word Analysis
- Top words by week/month
- Log-odds scoring
- Handles negations and noise filtering
