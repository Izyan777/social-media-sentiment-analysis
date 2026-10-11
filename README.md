# 🇵🇰 Pakistani Social Media Content Detective

**Business question:** *What type of social-media content gets Pakistani audiences to engage, and what should a Pakistani brand do next?*

We act as junior digital-marketing data scientists. The project has two connected parts:

| Part | What it does | Data | Status |
|---|---|---|---|
| **A. Pakistani sentiment + improvement engine** | Reads Pakistani comments (Roman Urdu / English), predicts sentiment, finds which business themes (quality, price, delivery, trust ...) drive negativity, and turns that into an improvement plan | **Real:** ~20k human-labelled Roman Urdu comments (e-commerce reviews, Facebook, Twitter) | ✅ Done and reproducible |
| **B. Engagement detective** | Answers: which platform, content type, hashtags, sentiment, time/day, audience segment earn the most engagement, then writes a "what to post next" summary | **Needs a Pakistani engagement file** (likes/shares/comments) - see below | ✅ Code done and tested; ⚠️ waiting for the Pakistani engagement dataset |

> **Honest scope note.** The starter repository provided for this course contains **no Pakistani rows** (12 countries, Pakistan is not one of them), and its posts are templated sentences. Part A therefore uses a real public Pakistani dataset. Part B's code is finished and tested, but the Pakistani engagement numbers must come from a Pakistani engagement dataset that the team adds (instructions below). We did **not** invent engagement numbers.

---

## Results (Part A, real Pakistani data)

**Data cleaning** (`data/processed/cleaning_log.json`): 20,229 raw rows → **19,328** clean unique comments (dropped 1 invalid label, 220 junk/empty texts, 209 rows whose identical text had conflicting labels, 471 duplicates). Classes: Neutral 8,319 · Positive 5,780 · Negative 5,229.

**Sentiment model**: TF-IDF (word 1-2 grams + character 2-5 grams) → Linear SVM, tuned with 5-fold cross-validation on the training set only, then scored once on a held-out 20% test set.

| | Macro-F1 | Accuracy |
|---|---|---|
| Majority-class baseline | 0.201 | - |
| Complement Naive Bayes (CV) | 0.630 | - |
| Logistic Regression (CV) | 0.645 | - |
| **Linear SVM (held-out test, 3,866 comments)** | **0.681** | **0.688** |

Per-class F1 on the test set: Negative 0.64 · Neutral 0.72 · Positive 0.68. See `reports/figures/02_confusion_matrix.png`.

**Which topics make Pakistani audiences unhappy?** (`reports/figures/04_theme_negativity.png`, `reports/theme_report.csv`)
Overall, 27% of comments are negative. Themes with at least 30 mentions:

| Theme | Comments | % negative | Lift | p (raw) | p (Bonferroni-corrected) |
|---|---|---|---|---|---|
| Product quality | 232 | 33% | ×1.23 | 0.041 | 0.165 |
| Delivery & logistics | 32 | 34% | ×1.27 | 0.46 | 1.0 |
| Taste & food | 35 | 29% | ×1.06 | 0.99 | 1.0 |
| Price & value | 116 | 29% | ×1.08 | 0.66 | 1.0 |

**Plain reading:** product quality is the most-discussed commerce topic and leans negative, but with four themes tested **no theme is statistically distinguishable from the average after correcting for multiple comparisons**. This corpus is too small and too mixed-domain to name "the" pain point with confidence. That is a real finding about the data, not a bug; a larger brand-specific comment set (Part B / your own scraped sample) is what would sharpen it. The generated action plan is in `reports/recommendations_roman_urdu_corpus.md`.

### A correction we made during development (important for honesty)
Our first keyword lexicon produced much stronger-looking numbers (quality 53% negative, trust 50%, price 37%, all "significant"). An audit of random matches showed those numbers were **inflated** for two reasons: (1) we had used sentiment words such as *kharab, bakwas, nakli, dhoka* as theme keywords, so those themes were negative by construction (circular reasoning); (2) several keywords matched non-commerce text (*box* → "box office", *seal* → school sealing, *service* → "Civil Service", *thori der* → "a little while"). We rebuilt the lexicon from topic-neutral nouns only, removed ambiguous words, and added a Bonferroni correction. The honest numbers above are smaller. Complaint words are still used by the brand-comment tool (`src/recommend.py`), where we only want to spot problem topics in someone's own comments and make no statistical claim.

**Brand demo:** `python -m src.recommend --input data/sample/demo_brand_comments.csv` reads a brand's comments and writes an improvement report (`reports/brand_comment_report.md`). The demo file contains 40 *hand-written illustrative* comments (not real customers).

## What we tried that did not help (reported on purpose)

- Adding the dataset's positive/negative word lexicons as extra features: +0.001 macro-F1, within noise → not kept.
- Wider regularisation grid: confirmed C ≈ 0.1 is the optimum (it was at the edge of the first grid, so we checked).

## Limitations - please read

1. **Mixed-domain training data.** The corpus mixes e-commerce reviews with news, politics and general Facebook/Twitter chatter, and has no product/brand column. Theme statistics describe *this corpus*, not a specific brand.
2. **The model under-detects "factual" complaints.** On our 40 hand-written e-commerce comments (a small, easy, self-written test - not a benchmark) it scored 35/40, and its misses were complaints with no emotional words (late delivery, refund pending, website hanging) labelled Neutral. Delivery and service problems are therefore probably under-counted as negative.
3. **Keyword themes are approximate.** Theme tagging uses a transparent hand-built keyword list (`src/insights.py`). After the rebuild, our manual read of ~10 random matches per theme found mostly genuine e-commerce comments, but some noise remains (e.g. *mazay* as slang, *dabba* as "container"). We have **not** measured precision formally: `reports/theme_audit_to_label.csv` is a sheet for the team to label ~70 random matches (fill the `relevant_yes_no` column) and report real precision per theme. Themes with fewer than 30 mentions (customer service, packaging, trust, app/payment, network) are too rare in this corpus to analyse.
4. **Neutral is a catch-all class** (tags, news-style statements), so it is the hardest and noisiest label.
5. **Part B has no Pakistani engagement results yet.** The only engagement run in this repo (`reports/engagement_summary_demo_sir_dataset.md`) is a code test on the starter dataset and is clearly marked *not Pakistani data*.

## Part B: plugging in a Pakistani engagement dataset

Needed: a CSV with at least one of **likes / shares / comments** per post, ideally also platform, timestamp, text, hashtags, content type (reel/video/image), and city/province.
Candidate sources to check (we could not download these from our environment, so verify the columns and licence yourself): IEEE DataPort "Pakistan Popular YouTube Videos", Kaggle social-media engagement datasets with a country column, or a small sample you collect by hand from public Pakistani brand pages (record likes/comments/shares/post time yourself).

```bash
python -m src.engagement --input data/raw/your_file.csv --country Pakistan --tz Asia/Karachi \
    --map likes=like_count,shares=share_count,comments=comment_count,text=caption,content_type=video_type
```

Any analysis whose columns are missing is skipped and listed in the summary. If your file has text but no sentiment column, sentiment is predicted with our trained Pakistani model. Engagement is heavy-tailed, so we compare **medians** and use **Kruskal-Wallis** tests; the heatmap's colour scale starts at zero so small differences are not exaggerated.

## How to run

```bash
pip install -r requirements.txt
python -m src.run_all        # download -> clean -> train -> themes -> recommendations (about 3 minutes)
```

Or open `notebooks/Pakistan_Content_Detective.ipynb` in Google Colab. Everything uses fixed random seeds, so numbers reproduce exactly.

## Repository layout

```
src/
  config.py          paths and constants
  download_data.py   fetches the public Roman Urdu dataset (git clone; not re-hosted here)
  preprocess.py      cleaning + Roman Urdu spelling normalisation
  train_sentiment.py model comparison, tuning, evaluation, figures
  insights.py        themes -> pain points -> recommended actions
  recommend.py       improvement report for any brand's comment CSV
  engagement.py      platform / content / sentiment / hashtag / time / segment analysis
  viz.py             one shared chart style
  run_all.py         the whole pipeline
data/sample/         hand-written demo comments
reports/             metrics, figures, recommendations, audit sheet
notebooks/           Colab notebook
```

## Data sources, credit and licence

- **Roman Urdu sentiment dataset:** compiled by Zareen Sharf (UCI *Roman Urdu Data Set*); GitHub copy [Smat26/Roman-Urdu-Dataset](https://github.com/Smat26/Roman-Urdu-Dataset), **GPL-3.0**. It is downloaded at run time, not stored in this repo. If you redistribute data derived from it, keep GPL-3.0 terms.
- **Course starter repository:** [mrudukp/social-media-sentiment-analysis](https://github.com/mrudukp/social-media-sentiment-analysis) (schema reference and engagement-analysis idea).
- The example comments quoted in the reports are short excerpts from the public dataset above.
