"""From sentiment to action: which themes make Pakistani customers unhappy?

1. Tag every comment with business themes (delivery, price, quality, service, ...) using a
   hand-built Roman Urdu + English keyword lexicon.
2. For each theme measure how often it is mentioned, how negative those comments are
   compared with the overall rate (lift), and whether the difference is statistically real
   (chi-square test).
3. Turn the ranked "pain points" into concrete improvement actions.

The lexicon is intentionally transparent: anyone can open THEMES, read it and extend it.
"""
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src import viz

# Keywords are matched on *normalised* text (see preprocess.normalize_text), as whole words
# (multi-word phrases are allowed).
#
# DESIGN RULES (learned from auditing random matches - see reports/theme_audit_to_label.csv):
#  1. THEMES use topic-neutral nouns only. Sentiment-laden words ("kharab", "bakwas", "nakli",
#     "dhoka") are NOT here, otherwise a theme would look negative by construction (circular).
#  2. Words that mostly mean something else in this mixed news/social corpus were removed:
#     "box" (box office), "seal" (school sealing), "service"/"staff"/"replace" (Civil Service,
#     cricket replacements), "der"/"dair" ("thori der" = a little while), "pohanch" (reached),
#     "app" (= "aap", you), "late" (also marks a deceased person), "speed" (cricket).
THEMES: dict[str, list[str]] = {
    "Delivery & logistics": [
        "delivery", "deliver", "delivered", "parcel", "courier", "tcs", "leopard", "rider",
        "shipping", "dispatch", "tracking", "delay", "delayed",
    ],
    "Price & value": [
        "price", "qeemat", "mehnga", "mehngai", "expensive", "costly", "discount", "sasta", "sasti", "sastay",
    ],
    "Product quality": ["quality", "material", "kapra", "kapray", "stitching", "size", "fitting", "colour", "color", "product"],
    "Trust & authenticity": ["asli", "original", "guarantee", "warranty", "bharosa", "genuine", "authentic"],
    "Customer service": [
        "customer service", "customer care", "customer support", "helpline", "complaint", "shikayat",
        "refund", "replacement", "bartao", "behaviour", "behavior",
    ],
    "Packaging": ["packing", "packaging", "dabba"],
    # NOTE: bare "app" is NOT a keyword - in Roman Urdu "app" is usually a typo of "aap" (you).
    "App / website / payment": ["application", "website", "login", "payment", "checkout", "cod"],
    "Network & internet": ["network", "signal", "internet", "coverage", "wifi", "jazz", "zong", "ufone", "telenor"],
    "Taste & food": ["taste", "mazedar", "mazay", "zaiqa", "tasty", "biryani", "burger", "pizza"],
}

# Complaint words. Used ONLY by the brand-comment tool (src/recommend.py), where we just want to
# spot a problem topic in someone's own comments - never for the corpus-level negativity statistics.
COMPLAINT_TERMS: dict[str, list[str]] = {
    "Delivery & logistics": ["late", "deri"],
    "Price & value": ["rs", "paisay", "paise", "paisa", "rupay", "rupee", "rupees"],
    "Product quality": ["kharab", "ghatiya", "bakwas", "damaged", "toota", "tuta", "tooti", "broken", "defective", "kamzor"],
    "Trust & authenticity": ["nakli", "dhoka", "jhoot", "jhoota", "trust"],
    "Customer service": ["support", "return", "rude", "response", "jawab"],
    "Packaging": ["box", "sealed"],
    "App / website / payment": ["error", "hang", "crash", "online"],
    "Network & internet": ["speed"],
    "Taste & food": ["khana", "food"],
}

ACTIONS: dict[str, str] = {
    "Delivery & logistics": (
        "Publish realistic delivery windows, add live parcel tracking, send proactive delay messages, "
        "and review courier partners by city. Track on-time delivery rate weekly."
    ),
    "Price & value": (
        "Reply to price complaints with clear value (what is included, warranty, bundles). Test "
        "targeted discounts or smaller pack sizes for price-sensitive cities and the sale-season calendar."
    ),
    "Product quality": (
        "Run a quality audit on the most-complained items, add real customer photos and exact "
        "size/material details to listings, and tighten supplier QA before dispatch."
    ),
    "Trust & authenticity": (
        "Show proof of originality (authorised-dealer badge, warranty card, verified seller). "
        "Fast-track any 'fake/duplicate' complaint and publish the outcome."
    ),
    "Customer service": (
        "Set a response-time promise (e.g. under 24 h), give agents Roman Urdu/Urdu templates, "
        "and make refund/return steps simple and visible. Track first-response time and resolution rate."
    ),
    "Packaging": (
        "Improve protective packaging for fragile items and use sealed, branded boxes so customers "
        "can tell nothing was opened or swapped."
    ),
    "App / website / payment": (
        "Fix the most reported checkout/payment errors first, keep Cash-on-Delivery prominent, and "
        "monitor crash/error rates after every release."
    ),
    "Network & internet": (
        "Prioritise coverage and speed fixes in the worst-rated areas and communicate outages quickly. "
        "Offer clear package details so expectations match real speeds."
    ),
    "Taste & food": (
        "Keep recipes and portion size consistent across outlets, and collect branch-level feedback so "
        "problem kitchens are fixed quickly."
    ),
}


def _compile(words: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b")


_PATTERNS = {theme: _compile(words) for theme, words in THEMES.items()}
_PATTERNS_WITH_COMPLAINTS = {theme: _compile(THEMES[theme] + COMPLAINT_TERMS.get(theme, [])) for theme in THEMES}


def tag_themes(clean_text: pd.Series, include_complaints: bool = False) -> pd.DataFrame:
    """Boolean matrix: one column per theme, True if the comment mentions it."""
    pats = _PATTERNS_WITH_COMPLAINTS if include_complaints else _PATTERNS
    return pd.DataFrame({t: clean_text.str.contains(p) for t, p in pats.items()}, index=clean_text.index)


def theme_report(
    df: pd.DataFrame, text_col: str = "clean_text", label_col: str = "label", min_mentions: int = 30, include_complaints: bool = False
) -> pd.DataFrame:
    """Rank themes by how much negativity they carry. Needs a sentiment label per comment."""
    tags = tag_themes(df[text_col], include_complaints)
    is_neg = (df[label_col] == "Negative").to_numpy()
    base_neg = is_neg.mean()
    rows = []
    for theme in THEMES:
        m = tags[theme].to_numpy()
        n = int(m.sum())
        if n < min_mentions:
            continue
        neg_in = int((m & is_neg).sum())
        table = [[neg_in, n - neg_in], [int((~m & is_neg).sum()), int((~m & ~is_neg).sum())]]
        p = float(chi2_contingency(table)[1])
        rows.append(
            {
                "theme": theme,
                "mentions": n,
                "share_of_all_comments": n / len(df),
                "negative_comments": neg_in,
                "negative_rate": neg_in / n,
                "lift_vs_overall": (neg_in / n) / base_neg,
                "p_value": p,
            }
        )
    rep = pd.DataFrame(rows)
    if rep.empty:
        return rep
    # We test several themes at once, so correct for multiple comparisons (Bonferroni) - otherwise
    # one "significant" result per ~20 tests would appear by pure chance.
    rep["p_adjusted"] = (rep["p_value"] * len(rep)).clip(upper=1.0)
    rep["significantly_more_negative"] = (rep["lift_vs_overall"] > 1) & (rep["p_adjusted"] < 0.05)
    return rep.sort_values("negative_comments", ascending=False).reset_index(drop=True)


def example_comments(
    df: pd.DataFrame, theme: str, text_col: str = "text", label_col: str = "label", k: int = 3, include_complaints: bool = False
) -> list[str]:
    tags = tag_themes(df["clean_text"], include_complaints)
    sub = df[tags[theme] & (df[label_col] == "Negative")]
    sub = sub[sub[text_col].str.len().between(25, 140)]
    return sub.sample(min(k, len(sub)), random_state=7)[text_col].str.strip().tolist()


def plot_theme_pain(rep: pd.DataFrame, base_neg: float) -> None:
    viz.setup()
    r = rep.sort_values("negative_rate")
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(r) + 1.8))
    y = np.arange(len(r))
    colors = [viz.RED if s else viz.GRAY for s in r["significantly_more_negative"]]
    ax.barh(y, r["negative_rate"], color=colors, height=0.55, zorder=2)
    ax.axvline(base_neg, color=viz.INK_2, lw=1.2, ls="--", zorder=1)
    ax.set_yticks(y, r["theme"])
    for yi, (nr, n, sig) in enumerate(zip(r["negative_rate"], r["mentions"], r["significantly_more_negative"])):
        tag = "  ▲ significantly more negative" if sig else ""
        ax.text(max(nr, base_neg) + 0.012, yi, f"{nr:.0%}  (n={n}){tag}", va="center", fontsize=9, color=viz.INK, zorder=3)
    ax.set_xlim(0, max(0.85, r["negative_rate"].max() + 0.35))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Share of comments mentioning the theme that are negative")
    if r["significantly_more_negative"].any():
        ax.set_title("Which topics make Pakistani audiences unhappy?")
    else:
        ax.set_title("Negativity by topic: none stands out statistically (yet)")
    ax.grid(axis="y", visible=False)
    viz.finish(
        fig,
        "04_theme_negativity",
        f"Dashed line = overall negative rate ({base_neg:.0%}). Red = significantly more negative than average "
        "(chi-square, Bonferroni-corrected p < 0.05); gray = not distinguishable from average at this sample size.",
    )


def recommendations_markdown(rep: pd.DataFrame, df: pd.DataFrame | None = None, include_complaints: bool = False) -> str:
    """Human-readable improvement plan built from the ranked themes."""
    lines = ["# AI-assisted improvement recommendations", ""]
    if rep.empty:
        return "\n".join(lines + ["Not enough theme mentions to make recommendations."])
    flagged = rep[rep["significantly_more_negative"]]
    lines += [
        "Priorities are ranked by **how many negative comments** each theme accounts for. "
        "Themes marked ▲ are *statistically* more negative than the overall average "
        "(chi-square test, Bonferroni-corrected for the number of themes tested, p < 0.05).",
        "",
    ]
    for i, row in enumerate(rep.itertuples(), 1):
        flag = " ▲" if row.significantly_more_negative else ""
        lines += [
            f"## {i}. {row.theme}{flag}",
            f"- Mentioned in **{row.mentions}** comments ({row.share_of_all_comments:.1%} of all); "
            f"**{row.negative_rate:.0%}** negative (lift ×{row.lift_vs_overall:.2f}, p={row.p_value:.3g}, "
            f"corrected p={row.p_adjusted:.3g}).",
            f"- **Action:** {ACTIONS[row.theme]}",
        ]
        if df is not None and "text" in df.columns:
            ex = example_comments(df, row.theme, include_complaints=include_complaints)
            if ex:
                lines.append("- Example negative comments:")
                lines += [f"  - “{e}”" for e in ex]
        lines.append("")
    if flagged.empty:
        lines.append("_No theme is significantly more negative than average in this sample._")
    return "\n".join(lines)
