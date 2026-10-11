"""Run the whole project end to end:   python -m src.run_all

1. download the public Roman Urdu dataset   2. clean + normalise
3. train / evaluate the sentiment model     4. theme analysis + improvement recommendations
5. brand-comment demo                       6. export a labelling sheet to audit the theme lexicon
"""
import pandas as pd

from src import download_data, insights, recommend, train_sentiment
from src.config import CLEAN_CSV, REPORTS_DIR, SAMPLE_DIR, ensure_dirs
from src.preprocess import load_and_clean


def export_theme_audit(df: pd.DataFrame, per_theme: int = 8) -> None:
    """Sample keyword matches so the TEAM can label them and measure the lexicon's precision."""
    tags = insights.tag_themes(df["clean_text"])
    rows = []
    for theme in insights.THEMES:
        sub = df[tags[theme]]
        for text in sub.sample(min(per_theme, len(sub)), random_state=11)["text"]:
            rows.append({"theme": theme, "text": str(text).strip(), "relevant_yes_no": ""})
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "theme_audit_to_label.csv", index=False)


def main() -> None:
    ensure_dirs()
    print("== 1/6 download ==")
    download_data.main()
    print("== 2/6 clean ==")
    df, log = load_and_clean()
    print(log)
    print("== 3/6 train sentiment model ==")
    train_sentiment.main()
    print("== 4/6 themes & recommendations ==")
    df = pd.read_csv(CLEAN_CSV)
    rep = insights.theme_report(df)
    rep.to_csv(REPORTS_DIR / "theme_report.csv", index=False)
    insights.plot_theme_pain(rep, (df["label"] == "Negative").mean())
    (REPORTS_DIR / "recommendations_roman_urdu_corpus.md").write_text(insights.recommendations_markdown(rep, df), encoding="utf-8")
    print(rep.round(3).to_string())
    print("== 5/6 brand comment demo ==")
    recommend.run(str(SAMPLE_DIR / "demo_brand_comments.csv"), "comment")
    print("== 6/6 theme audit sheet ==")
    export_theme_audit(df)
    print("\nDone. See the reports/ folder.")


if __name__ == "__main__":
    main()
