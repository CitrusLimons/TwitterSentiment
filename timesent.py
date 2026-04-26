import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

CSV_FILE = Path(r"D:\downloads\BigData\training.1600000.processed.noemoticon.csv")
OUTDIR = Path(r"D:\downloads\BigData\output")
OUTDIR.mkdir(parents=True, exist_ok=True)

COLS = ["sentiment", "id", "date", "query", "user", "text"]
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

KNOWN_EVENTS = {
    "2009-04-06": "L'Aquila earthquake",
    "2009-04-09": "Easter",
    "2009-04-15": "Tax Day (US)",
    "2009-05-01": "Swine flu pandemic",
    "2009-05-25": "Memorial Day",
    "2009-06-04": "Tiananmen anniv.",
    "2009-06-06": "D-Day anniv.",
    "2009-06-09": "NBA Finals begin",
    "2009-06-12": "Iran election",
    "2009-06-13": "Iran protests peak",
    "2009-06-21": "Father's Day",
    "2009-06-25": "Michael Jackson dies",
    "2009-06-26": "Farrah Fawcett dies",
}

MIN_TWEETS_PER_DAY = 500

tqdm.pandas(desc="Parsing dates")


# ────────────────────────────────────────────────────────────
# DATA LOADING
# ────────────────────────────────────────────────────────────

def parse_date(date_str):
    try:
        s = str(date_str).replace(" PDT ", " ").replace(" PST ", " ")
        dt = datetime.strptime(s, "%a %b %d %H:%M:%S %Y")
        return pd.Series({
            "day":         dt.strftime("%Y-%m-%d"),
            "month":       dt.strftime("%Y-%m"),
            "weekday_num": dt.weekday(),
            "weekday":     WEEKDAY_NAMES[dt.weekday()],
            "is_weekend":  1 if dt.weekday() >= 5 else 0,
            "hour":        dt.hour,
        })
    except Exception:
        return pd.Series({k: pd.NA for k in ["day", "month", "weekday_num", "weekday", "is_weekend", "hour"]})


def load_data():
    print("Loading data...")
    df = pd.read_csv(CSV_FILE, encoding="latin-1", header=None, names=COLS)
    print(f"  Raw rows: {len(df):,}")
    df = df[df["sentiment"].isin([0, 4])].copy()
    df["label"] = df["sentiment"].map({0: 0, 4: 1})
    parsed = df["date"].progress_apply(parse_date)
    df = pd.concat([df.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
    df = df.dropna(subset=["day", "month", "weekday_num", "weekday", "is_weekend", "hour"]).copy()
    df["weekday_num"] = df["weekday_num"].astype(int)
    df["is_weekend"]  = df["is_weekend"].astype(int)
    df["hour"]        = df["hour"].astype(int)
    print(f"  Usable rows: {len(df):,}")
    return df


# ────────────────────────────────────────────────────────────
# AGGREGATION
# ────────────────────────────────────────────────────────────

def get_valid_days(df):
    day_counts = df.groupby("day")["label"].count()
    return day_counts[day_counts >= MIN_TWEETS_PER_DAY].index


def build_weekday_summary(df):
    valid_days = get_valid_days(df)
    df_valid = df[df["day"].isin(valid_days)]

    s = df_valid.groupby("weekday_num")["label"].agg(
        total_tweets="count",
        positive_tweets="sum",
        positive_ratio="mean"
    ).reset_index()

    full_week = pd.DataFrame({"weekday_num": range(7), "weekday": WEEKDAY_NAMES})
    s = full_week.merge(s, on="weekday_num", how="left")
    s["total_tweets"]    = s["total_tweets"].fillna(0).astype(int)
    s["positive_tweets"] = s["positive_tweets"].fillna(0).astype(int)
    s["negative_tweets"] = s["total_tweets"] - s["positive_tweets"]
    s["negative_ratio"]  = 1 - s["positive_ratio"]
    return s.sort_values("weekday_num").reset_index(drop=True)


def build_hourly_summary(df):
    valid_days = get_valid_days(df)
    df_valid = df[df["day"].isin(valid_days)]

    h = df_valid.groupby("hour")["label"].agg(
        total_tweets="count",
        positive_tweets="sum",
        positive_ratio="mean"
    ).reset_index()
    full_hours = pd.DataFrame({"hour": range(24)})
    h = full_hours.merge(h, on="hour", how="left")
    h["positive_ratio"] = h["positive_ratio"].fillna(h["positive_ratio"].mean())
    h["total_tweets"]   = h["total_tweets"].fillna(0).astype(int)
    h["negative_ratio"] = 1 - h["positive_ratio"]
    return h


def build_hour_weekday_heatmap(df):
    valid_days = get_valid_days(df)
    df_valid = df[df["day"].isin(valid_days)]
    return (
        df_valid.groupby(["weekday_num", "hour"])["label"]
        .mean()
        .reset_index()
        .rename(columns={"label": "positive_ratio"})
    )


# ────────────────────────────────────────────────────────────
# TREND ANALYSIS
# ────────────────────────────────────────────────────────────

def compute_trend(series):
    x = np.arange(len(series))
    mask = series.notna()
    if mask.sum() < 3:
        return 0.0, 0.0, 1.0
    slope, intercept, r, p, _ = stats.linregress(x[mask], series.values[mask])
    return slope, r**2, p


def rolling_correlation(a, b, window=7):
    return a.rolling(window).corr(b)


# ────────────────────────────────────────────────────────────
# PLOTS
# ────────────────────────────────────────────────────────────

def plot_rolling_correlation(df, outdir):
    """Rolling correlation built directly from raw data, not day_df."""
    valid_days = get_valid_days(df)
    day_summary = df[df["day"].isin(valid_days)].groupby("day")["label"].agg(
        total_tweets="count",
        positive_ratio="mean"
    ).reset_index().sort_values("day")
    day_summary["day_dt"] = pd.to_datetime(day_summary["day"])

    corr = rolling_correlation(
        day_summary["total_tweets"].reset_index(drop=True),
        day_summary["positive_ratio"].reset_index(drop=True),
        window=7
    )
    corr_vals = corr.values

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axhline(0, color="black", linewidth=1)
    ax.fill_between(day_summary["day_dt"], corr_vals,
                    where=(~np.isnan(corr_vals)) & (corr_vals >= 0),
                    color="seagreen", alpha=0.4, label="Vol ↑ → more positive")
    ax.fill_between(day_summary["day_dt"], corr_vals,
                    where=(~np.isnan(corr_vals)) & (corr_vals < 0),
                    color="firebrick", alpha=0.4, label="Vol ↑ → more negative")
    ax.plot(day_summary["day_dt"], corr_vals, color="gray", linewidth=1)
    ax.set_title("Rolling 7-day Correlation: Tweet Volume vs Positive Ratio")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-1, 1)

    tick_idx = list(range(0, len(day_summary), 5))
    if tick_idx[-1] != len(day_summary) - 1:
        tick_idx.append(len(day_summary) - 1)
    ax.set_xticks([day_summary["day_dt"].iloc[i] for i in tick_idx])
    ax.set_xticklabels(
        [day_summary["day_dt"].iloc[i].strftime("%Y-%m-%d") for i in tick_idx],
        rotation=45, ha="right"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "rolling_volume_sentiment_correlation.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: rolling_volume_sentiment_correlation.png")


def plot_hourly_sentiment(hourly_df, outdir):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(hourly_df["hour"], hourly_df["total_tweets"],
            color="lightskyblue", alpha=0.7, label="Volume")
    ax1.set_xlabel("Hour of Day (UTC)")
    ax1.set_ylabel("Tweet Count", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_xticks(range(24))

    ax2 = ax1.twinx()
    ax2.plot(hourly_df["hour"], hourly_df["positive_ratio"],
             color="darkorange", linewidth=2.5, marker="o", label="Positive ratio")
    ax2.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.4)
    ax2.set_ylabel("Positive Ratio", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_ylim(0.3, 0.7)

    plt.title("Tweet Volume and Sentiment by Hour of Day (UTC)")
    fig.tight_layout()
    plt.savefig(outdir / "sentiment_by_hour.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: sentiment_by_hour.png")


def plot_heatmap(heat_df, outdir):
    pivot = heat_df.pivot(index="weekday_num", columns="hour", values="positive_ratio")
    pivot = pivot.reindex(index=range(7), columns=range(24))
    col_means = pivot.mean()
    for col in pivot.columns:
        pivot[col] = pivot[col].fillna(col_means[col])

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=0.35, vmax=0.65, origin="upper")
    ax.set_yticks(range(7))
    ax.set_yticklabels(WEEKDAY_NAMES)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=90, fontsize=7)
    ax.set_title("Positive Sentiment Ratio Heatmap  (Hour × Day of Week)")
    plt.colorbar(im, ax=ax, label="Positive Ratio")
    plt.tight_layout()
    plt.savefig(outdir / "heatmap_hour_weekday.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: heatmap_hour_weekday.png")


def plot_weekday_profile(weekday_df, outdir):
    fig, ax = plt.subplots(figsize=(10, 5))

    ratio = weekday_df["positive_ratio"].fillna(0)
    colors = ["mediumpurple" if r > 0.5 else "salmon" for r in ratio]
    bars = ax.bar(weekday_df["weekday"], ratio, color=colors)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Neutral (0.5)")
    mean_ratio = ratio[ratio > 0].mean()
    ax.axhline(mean_ratio, color="steelblue", linestyle=":", linewidth=1,
               label=f"Dataset mean ({mean_ratio:.3f})")

    for bar, row in zip(bars, weekday_df.itertuples()):
        label = f"n={row.total_tweets:,}" if row.total_tweets > 0 else "no data"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                label, ha="center", va="bottom", fontsize=7, color="gray")

    valid_ratios = ratio[ratio > 0]
    y_min = max(0.0, valid_ratios.min() - 0.05) if len(valid_ratios) else 0.4
    y_max = min(1.0, valid_ratios.max() + 0.06) if len(valid_ratios) else 0.6
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Positive Sentiment Ratio by Day of Week  (days ≥{MIN_TWEETS_PER_DAY} tweets only)")
    ax.set_ylabel("Positive Ratio")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "sentiment_by_weekday.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: sentiment_by_weekday.png")


# ────────────────────────────────────────────────────────────
# INSIGHT REPORT
# ────────────────────────────────────────────────────────────

def print_insight_report(df, hourly_df, weekday_df):
    print("\n" + "═" * 60)
    print("  PATTERN INSIGHT REPORT")
    print("═" * 60)

    peak_pos_hour = int(hourly_df.loc[hourly_df["positive_ratio"].idxmax(), "hour"])
    peak_neg_hour = int(hourly_df.loc[hourly_df["positive_ratio"].idxmin(), "hour"])
    peak_vol_hour = int(hourly_df.loc[hourly_df["total_tweets"].idxmax(), "hour"])
    print(f"\n[HOURLY] Most positive hour: {peak_pos_hour:02d}:00 UTC")
    print(f"         Most negative hour:  {peak_neg_hour:02d}:00 UTC")
    print(f"         Highest volume hour: {peak_vol_hour:02d}:00 UTC")

    valid_wd = weekday_df[weekday_df["positive_ratio"].notna() & (weekday_df["total_tweets"] > 0)]
    if not valid_wd.empty:
        best_day  = valid_wd.loc[valid_wd["positive_ratio"].idxmax(), "weekday"]
        worst_day = valid_wd.loc[valid_wd["positive_ratio"].idxmin(), "weekday"]
        print(f"\n[WEEKDAY] Most positive day: {best_day}")
        print(f"          Most negative day:  {worst_day}")

    valid_days = get_valid_days(df)
    day_summary = df[df["day"].isin(valid_days)].groupby("day")["label"].agg(
        total_tweets="count", positive_ratio="mean"
    ).reset_index().sort_values("day")
    day_summary["day_dt"] = pd.to_datetime(day_summary["day"])

    corr = rolling_correlation(
        day_summary["total_tweets"].reset_index(drop=True),
        day_summary["positive_ratio"].reset_index(drop=True), window=7
    )
    neg_corr_days = int((corr < -0.3).sum())
    pos_corr_days = int((corr >  0.3).sum())
    print(f"\n[VOL-SENTIMENT CORRELATION]")
    print(f"  Days where high volume → more negative: {neg_corr_days}")
    print(f"  Days where high volume → more positive: {pos_corr_days}")
    print(f"  Interpretation: {'Stress events drive more tweeting' if neg_corr_days > pos_corr_days else 'Positive events drive more tweeting'}")

    print("\n" + "═" * 60 + "\n")


# ────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────

def main():
    df = load_data()

    weekday_df = build_weekday_summary(df)
    hourly_df  = build_hourly_summary(df)
    heat_df    = build_hour_weekday_heatmap(df)

    weekday_df.to_csv(OUTDIR / "sentiment_by_weekday.csv", index=False)
    hourly_df.to_csv(OUTDIR / "sentiment_by_hour.csv",     index=False)
    print("CSVs saved.")

    print("Generating plots...")
    plt.style.use("ggplot")
    plot_rolling_correlation(df, OUTDIR)
    plot_hourly_sentiment(hourly_df, OUTDIR)
    plot_heatmap(heat_df, OUTDIR)
    plot_weekday_profile(weekday_df, OUTDIR)

    print_insight_report(df, hourly_df, weekday_df)


if __name__ == "__main__":
    main()