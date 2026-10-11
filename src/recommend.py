"""AI-assisted marketing recommendation for a brand's OWN comments.

Give it a CSV of comments (Roman Urdu / Urdu-in-Latin / English). It will:
  1. predict the sentiment of every comment with the trained Pakistani sentiment model,
  2. tag each comment with business themes (delivery, price, quality, service, ...),
  3. rank the themes by how much negativity they carry,
  4. write an improvement plan.

    python -m src.recommend --input data/sample/demo_brand_comments.csv --text-col comment
"""
import argparse

import joblib
import pandas as pd

from src import insights
from src.config import MODEL_PATH, REPORTS_DIR, ensure_dirs
from src.preprocess import normalize_text


def run(input_csv: str, text_col: str, out_md: str | None = None, min_mentions: int = 3) -> str:
    ensure_dirs()
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run:  python -m src.train_sentiment")
    df = pd.read_csv(input_csv)
    df = df[df[text_col].notna()].copy()
    df["text"] = df[text_col].astype(str)
    df["clean_text"] = df["text"].map(normalize_text)
    df["label"] = joblib.load(MODEL_PATH).predict(df["clean_text"])

    counts = df["label"].value_counts()
    # Brand tool: also spot complaint words (kharab, nakli, ...). Fine here because we only want to
    # surface problem topics in the brand's own comments, not to prove a statistical claim.
    rep = insights.theme_report(df, min_mentions=min_mentions, include_complaints=True)
    md = [
        "# Brand comment report",
        "",
        f"Comments analysed: **{len(df)}**  |  "
        + "  |  ".join(f"{k}: {counts.get(k, 0)} ({counts.get(k, 0) / len(df):.0%})" for k in ("Positive", "Neutral", "Negative")),
        "",
        "_Sentiment is predicted by our model (about 69% accurate on held-out Pakistani Roman Urdu data), "
        "so treat individual labels as estimates and the overall pattern as the signal. "
        "With small files, p-values are not meaningful - use the ranking as a guide only._",
        "",
        insights.recommendations_markdown(rep, df, include_complaints=True),
    ]
    text = "\n".join(md)
    out = out_md or str(REPORTS_DIR / "brand_comment_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--text-col", default="comment")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-mentions", type=int, default=3)
    a = ap.parse_args()
    print(run(a.input, a.text_col, a.out, a.min_mentions))


if __name__ == "__main__":
    main()
