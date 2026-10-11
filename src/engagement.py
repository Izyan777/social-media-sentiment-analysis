"""Engagement analysis: "What type of content gets Pakistani audiences to engage?"

Works on ANY social-media CSV. Point it at a file, tell it which of your columns mean what,
and it answers sir's questions: platform, content type, sentiment, hashtags, time/day,
audience segment - then writes a plain-English "what to post next" summary.

    python -m src.engagement --input data/raw/pakistan_posts.csv --country Pakistan \
        --map likes=like_count,shares=share_count,comments=comment_count,text=caption

Standard column names it understands (after --map renaming):
    text, platform, timestamp, likes, shares, comments, followers, hashtags,
    country, sentiment, verified, content_type, region (city / province / segment)

Any analysis whose columns are missing is skipped and listed in the summary - nothing is faked.

Engagement counts are heavy-tailed (a few viral posts), so we compare MEDIANS and use
non-parametric tests (Kruskal-Wallis) instead of means and t-tests.
"""
import argparse
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal

from src import viz
from src.config import MODEL_PATH, REPORTS_DIR, ensure_dirs
from src.preprocess import normalize_text

# Default mapping = the column names used by sir's original repository (standard -> source)
DEFAULT_MAP = {
    "shares": "retweets_shares",
    "followers": "followers_count",
}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAYPARTS = [(0, 6, "00-06 night"), (6, 12, "06-12 morning"), (12, 18, "12-18 afternoon"), (18, 24, "18-24 evening")]


def parse_map(s: str | None) -> dict:
    if not s:
        return {}
    return dict(item.split("=", 1) for item in s.split(",") if "=" in item)


def load_and_prepare(path: str, colmap: dict, country: str | None, tz: str | None) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    df = pd.read_csv(path)
    n0 = len(df)
    mapping = {**DEFAULT_MAP, **colmap}  # standard -> source
    df = df.rename(columns={src: std for std, src in mapping.items() if src in df.columns and std != src})

    # --- text/categorical hygiene ---
    for c in ("platform", "sentiment", "country", "content_type", "region"):
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip().str.title()
    if "followers" in df.columns:
        df["followers"] = pd.to_numeric(df["followers"].astype("string").str.replace(",", "", regex=False), errors="coerce").astype("float64")
    for c in ("likes", "shares", "comments"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    if "post_id" in df.columns:
        d = len(df)
        df = df.drop_duplicates("post_id")
        if d - len(df):
            notes.append(f"Dropped {d - len(df)} duplicate post_id rows.")

    # --- country filter (the "Pakistan only" requirement) ---
    if country and country.upper() != "ALL":
        if "country" not in df.columns:
            notes.append("No country column in the file - assuming every row is already from the target country.")
        else:
            df = df[df["country"].str.lower() == country.lower()]
            notes.append(f"Country filter '{country}': kept {len(df)} of {n0} rows.")
            if df.empty:
                raise SystemExit(f"No rows left after filtering country == {country!r}. Available: check your file's country column.")

    # --- engagement ---
    comps = [c for c in ("likes", "shares", "comments") if c in df.columns]
    if not comps:
        raise SystemExit("Need at least one of: likes, shares, comments (use --map to rename your columns).")
    neg = (df[comps] < 0).any(axis=1)
    if neg.any():
        notes.append(f"Dropped {int(neg.sum())} rows with negative engagement counts (invalid).")
        df = df[~neg]
    miss = df[comps].isna().sum().sum()
    if miss:
        notes.append(f"Filled {int(miss)} missing engagement counts with 0.")
    df[comps] = df[comps].fillna(0)
    df["engagement"] = df[comps].sum(axis=1)
    if "followers" in df.columns:
        ok = df["followers"] > 0
        df["engagement_per_1k_followers"] = np.where(ok, df["engagement"] / df["followers"].where(ok) * 1000, np.nan)

    # --- time ---
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=bool(tz))
        if tz:
            ts = ts.dt.tz_convert(tz)
            notes.append(f"Timestamps converted from UTC to {tz}.")
        df["timestamp"] = ts
        df = df[df["timestamp"].notna()].copy()
        df["day"] = pd.Categorical(df["timestamp"].dt.day_name().str[:3], categories=DAYS, ordered=True)
        df["hour"] = df["timestamp"].dt.hour
        df["daypart"] = pd.cut(df["hour"], bins=[0, 6, 12, 18, 24], right=False, labels=[p[2] for p in DAYPARTS])

    # --- hashtags ---
    if "hashtags" in df.columns:
        df["hashtag_list"] = df["hashtags"].fillna("").astype(str).map(lambda s: sorted(set(re.findall(r"#\w+", s.lower()))))

    # --- sentiment: use the file's own, or predict with our Pakistani Roman Urdu model ---
    if "text" in df.columns and MODEL_PATH.exists():
        if "sentiment" not in df.columns or df["sentiment"].isna().any():
            model = joblib.load(MODEL_PATH)
            need = df["sentiment"].isna() if "sentiment" in df.columns else pd.Series(True, index=df.index)
            if "sentiment" not in df.columns:
                df["sentiment"] = pd.NA
            df.loc[need, "sentiment"] = model.predict(df.loc[need, "text"].astype(str).map(normalize_text))
            notes.append(f"Predicted sentiment for {int(need.sum())} posts with the trained Roman Urdu model.")
    notes.append(f"Analysing {len(df)} posts. Engagement = {' + '.join(comps)}.")
    return df.reset_index(drop=True), notes


# ----------------------------- stats helpers -----------------------------

def group_table(df: pd.DataFrame, by: str, metric: str = "engagement", min_n: int = 15) -> tuple[pd.DataFrame, float | None]:
    g = df.groupby(by, observed=True)[metric].agg(["count", "median", "mean"]).rename(columns={"count": "n"})
    g = g[g["n"] >= min_n].sort_values("median", ascending=False)
    p = None
    groups = [x[metric].dropna().to_numpy() for _, x in df[df[by].isin(g.index)].groupby(by, observed=True)]
    if len(groups) >= 2:
        try:
            p = float(kruskal(*groups).pvalue)
        except ValueError:  # all values identical
            p = None
    return g, p


def verdict(p: float | None) -> str:
    if p is None:
        return "not enough data to test"
    return "statistically reliable difference" if p < 0.05 else "no statistically reliable difference (could be chance)"


# ----------------------------- charts -----------------------------

def _bars(g: pd.DataFrame, title: str, name: str, p: float | None, colors=None, xlabel="Median engagement per post") -> str:
    viz.setup()
    g = g.sort_values("median")
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(g) + 1.9))
    y = np.arange(len(g))
    cols = [colors.get(i, viz.BLUE) for i in g.index] if colors else viz.BLUE
    ax.barh(y, g["median"], color=cols, height=0.55)
    ax.set_yticks(y, [str(i) for i in g.index])
    for yi, (m, n) in enumerate(zip(g["median"], g["n"])):
        ax.text(m + g["median"].max() * 0.015, yi, f"{m:,.0f}  (n={n})", va="center", fontsize=9.5)
    ax.set_xlim(0, g["median"].max() * 1.25)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    sub = f"Kruskal-Wallis p = {p:.3g}: {verdict(p)}" if p is not None else None
    return viz.finish(fig, name, sub)


def plot_heatmap(df: pd.DataFrame, name: str) -> tuple[str, pd.DataFrame]:
    viz.setup()
    from matplotlib.colors import LinearSegmentedColormap

    cell = df.groupby(["day", "daypart"], observed=True)["engagement"].agg(["median", "count"]).reset_index()
    piv = cell.pivot(index="day", columns="daypart", values="median").reindex(DAYS)
    cnt = cell.pivot(index="day", columns="daypart", values="count").reindex(DAYS)
    masked = piv.where(cnt >= 10)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    cmap = LinearSegmentedColormap.from_list("blues", viz.SEQ_BLUES)
    # Colour scale starts at 0 on purpose: a truncated scale would make small differences look dramatic.
    im = ax.imshow(masked.to_numpy(dtype=float), cmap=cmap, aspect="auto", vmin=0)
    ax.set_xticks(range(piv.shape[1]), [str(c) for c in piv.columns])
    ax.set_yticks(range(len(DAYS)), DAYS)
    ax.grid(False)
    vmax = np.nanmax(masked.to_numpy(dtype=float)) if masked.notna().any().any() else 1
    for i in range(masked.shape[0]):
        for j in range(masked.shape[1]):
            v = masked.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center", color="white" if v > 0.55 * vmax else viz.INK, fontsize=10)
    fig.colorbar(im, ax=ax, label="Median engagement", shrink=0.8)
    ax.set_title("When do posts perform best? (day × time of day)")
    path = viz.finish(fig, name, "Cells with fewer than 10 posts are left blank to avoid noisy conclusions.")
    return path, masked


# ----------------------------- main analysis -----------------------------

def analyse(df: pd.DataFrame, tag: str, min_n: int = 15) -> dict:
    out = {"figures": [], "findings": [], "skipped": []}
    prefix = f"engagement_{tag}"

    if "platform" in df.columns:
        g, p = group_table(df, "platform", min_n=min_n)
        if len(g) >= 2:
            out["figures"].append(_bars(g, "Which platform earns the most engagement?", f"{prefix}/E1_platform", p))
            out["findings"].append(f"**Platform:** {g.index[0]} leads (median {g['median'].iloc[0]:,.0f} engagements/post vs "
                                   f"{g['median'].iloc[-1]:,.0f} for {g.index[-1]}) - {verdict(p)}.")
    else:
        out["skipped"].append("platform")

    if "content_type" in df.columns:
        g, p = group_table(df, "content_type", min_n=min_n)
        if len(g) >= 2:
            out["figures"].append(_bars(g, "Which content type earns the most engagement?", f"{prefix}/E2_content_type", p))
            out["findings"].append(f"**Content type:** {g.index[0]} performs best (median {g['median'].iloc[0]:,.0f}) - {verdict(p)}.")
    else:
        out["skipped"].append("content_type (no column - e.g. reel/video/image/text)")

    if "sentiment" in df.columns:
        g, p = group_table(df, "sentiment", min_n=min_n)
        if len(g) >= 2:
            out["figures"].append(_bars(g, "Which sentiment generates engagement?", f"{prefix}/E3_sentiment", p, colors=viz.SENTIMENT_COLORS))
            out["findings"].append(f"**Sentiment:** {g.index[0]} posts get the most engagement (median {g['median'].iloc[0]:,.0f}) - {verdict(p)}.")
    else:
        out["skipped"].append("sentiment")

    if "hashtag_list" in df.columns:
        ex = df.explode("hashtag_list").dropna(subset=["hashtag_list"])
        ex = ex[ex["hashtag_list"] != ""]
        if not ex.empty:
            g, p = group_table(ex, "hashtag_list", min_n=max(8, min_n // 2))
            g = g.head(10)
            if len(g) >= 3:
                out["figures"].append(_bars(g, "Top hashtags by median engagement", f"{prefix}/E4_hashtags", None))
                out["findings"].append("**Hashtags:** top performers are " + ", ".join(g.index[:3]) + " (compare with their sample sizes before acting).")
    else:
        out["skipped"].append("hashtags")

    if "daypart" in df.columns:
        path, masked = plot_heatmap(df, f"{prefix}/E5_time_heatmap")
        out["figures"].append(path)
        _, p_day = group_table(df, "day", min_n=min_n)
        stacked = masked.stack()
        if not stacked.empty:
            d, part = stacked.idxmax()
            out["findings"].append(f"**Timing:** the strongest slot is {d} {part} (median {stacked.max():,.0f}); "
                                   f"day-of-week overall: {verdict(p_day)}.")
    else:
        out["skipped"].append("time/day (no timestamp)")

    if "region" in df.columns:
        g, p = group_table(df, "region", min_n=min_n)
        if len(g) >= 2:
            out["figures"].append(_bars(g.head(10), "Which Pakistani audience segment is most responsive?", f"{prefix}/E6_region", p))
            out["findings"].append(f"**Audience segment:** {g.index[0]} is the most responsive (median {g['median'].iloc[0]:,.0f}) - {verdict(p)}.")
    else:
        out["skipped"].append("region/city/segment (no column)")

    if "followers" in df.columns and "verified" in df.columns:
        v = df.groupby("verified")["engagement_per_1k_followers"].median()
        out["findings"].append("**Verified vs not (engagement per 1k followers):** " + ", ".join(f"{k}: {x:,.1f}" for k, x in v.items()) + ".")
    return out


def write_summary(df: pd.DataFrame, notes: list[str], res: dict, tag: str, demo: bool) -> str:
    L = [f"# Engagement findings ({tag})", ""]
    if demo:
        L += ["> **DEMO RUN - NOT PAKISTANI DATA.** This run used a multi-country sample only to test that the code works. "
              "Do not quote these numbers as Pakistan results.", ""]
    L += ["## Data notes"] + [f"- {n}" for n in notes] + [""]
    L += ["## What the data says"] + [f"- {f}" for f in res["findings"]] + [""]
    L += ["## Suggested next post (rule-based from the findings above)"]
    L += ["Use the strongest platform, the best-performing sentiment/tone, the top hashtags with enough posts behind them, "
          "and the highest-engagement time slot - but only where the finding above says the difference is statistically reliable. "
          "Where it says 'could be chance', treat it as a hypothesis to A/B test, not a rule.", ""]
    if res["skipped"]:
        L += ["## Not analysed (missing columns)"] + [f"- {s}" for s in res["skipped"]] + [""]
    L += ["## Figures"] + [f"- `reports/figures/{p.split('/figures/')[-1]}`" for p in res["figures"]]
    text = "\n".join(L)
    (REPORTS_DIR / f"engagement_summary_{tag}.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--map", help="standard=your_column pairs, comma separated")
    ap.add_argument("--country", default="Pakistan", help="country to keep, or ALL")
    ap.add_argument("--tz", default=None, help="e.g. Asia/Karachi (use if timestamps are UTC)")
    ap.add_argument("--tag", default="pakistan")
    ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--demo", action="store_true", help="mark outputs as a non-Pakistani code test")
    a = ap.parse_args()
    ensure_dirs()
    df, notes = load_and_prepare(a.input, parse_map(a.map), a.country, a.tz)
    res = analyse(df, a.tag, a.min_n)
    print(write_summary(df, notes, res, a.tag, a.demo))


if __name__ == "__main__":
    main()
