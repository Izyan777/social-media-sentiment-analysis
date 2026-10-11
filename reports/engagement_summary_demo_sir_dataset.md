# Engagement findings (demo_sir_dataset)

> **DEMO RUN - NOT PAKISTANI DATA.** This run used a multi-country sample only to test that the code works. Do not quote these numbers as Pakistan results.

## Data notes
- Dropped 60 duplicate post_id rows.
- Dropped 8 rows with negative engagement counts (invalid).
- Filled 44 missing engagement counts with 0.
- Predicted sentiment for 33 posts with the trained Roman Urdu model.
- Analysing 2192 posts. Engagement = likes + shares + comments.

## What the data says
- **Platform:** Tiktok leads (median 57 engagements/post vs 24 for Linkedin) - statistically reliable difference.
- **Sentiment:** Positive posts get the most engagement (median 50) - statistically reliable difference.
- **Hashtags:** top performers are #ai, #climate, #gaming (compare with their sample sizes before acting).
- **Timing:** the strongest slot is Thu 06-12 morning (median 48); day-of-week overall: no statistically reliable difference (could be chance).
- **Verified vs not (engagement per 1k followers):** No: 22.5, Yes: 37.6.

## Suggested next post (rule-based from the findings above)
Use the strongest platform, the best-performing sentiment/tone, the top hashtags with enough posts behind them, and the highest-engagement time slot - but only where the finding above says the difference is statistically reliable. Where it says 'could be chance', treat it as a hypothesis to A/B test, not a rule.

## Not analysed (missing columns)
- content_type (no column - e.g. reel/video/image/text)
- region/city/segment (no column)

## Figures
- `reports/figures/engagement_demo_sir_dataset/E1_platform.png`
- `reports/figures/engagement_demo_sir_dataset/E3_sentiment.png`
- `reports/figures/engagement_demo_sir_dataset/E4_hashtags.png`
- `reports/figures/engagement_demo_sir_dataset/E5_time_heatmap.png`