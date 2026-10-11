"""Train and evaluate the Roman Urdu / Pakistani sentiment classifier.

Pipeline (all fitted inside sklearn Pipelines so no test information leaks):
    normalised text -> TF-IDF word 1-2 grams  +  TF-IDF character 2-5 grams -> linear classifier

Why character n-grams? Roman Urdu is spelled inconsistently ("acha", "achha", "accha"),
so sub-word features are much more robust than whole words alone.

Usage:  python -m src.train_sentiment
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src import viz
from src.config import LABELS, MODEL_PATH, RANDOM_STATE, REPORTS_DIR, ensure_dirs
from src.preprocess import load_and_clean


def make_features() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
        ]
    )


def make_pipeline(clf) -> Pipeline:
    return Pipeline([("features", make_features()), ("clf", clf)])


def candidate_models() -> dict:
    return {
        "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
        "Complement Naive Bayes": ComplementNB(alpha=0.3),
        "Logistic Regression": LogisticRegression(C=5, max_iter=2000, class_weight="balanced"),
        "Linear SVM": LinearSVC(C=0.4, class_weight="balanced"),
    }


def plot_model_comparison(scores: dict) -> None:
    viz.setup()
    names = list(scores)
    vals = [scores[n][0] for n in names]
    errs = [scores[n][1] for n in names]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, xerr=errs, color=viz.BLUE, height=0.5, ecolor=viz.INK_2, capsize=3)
    ax.set_yticks(y, names)
    for yi, v, e in zip(y, vals, errs):
        ax.text(v + e + 0.015, yi, f"{v:.3f}", va="center", fontsize=10, color=viz.INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1 (5-fold cross-validation on training set, mean ± std)")
    ax.set_title("Model comparison: classical ML beats the baseline by a wide margin")
    ax.grid(axis="y", visible=False)
    viz.finish(fig, "01_model_comparison")


def plot_confusion(cm: np.ndarray) -> None:
    viz.setup()
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blues", viz.SEQ_BLUES)
    ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            colour = "white" if cmn[i, j] > 0.5 else viz.INK
            ax.text(j, i, f"{cmn[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center", color=colour, fontsize=10.5)
    ax.set_title("Where the model gets confused (held-out test set)")
    viz.finish(fig, "02_confusion_matrix")


def top_terms(pipe: Pipeline, k: int = 14) -> dict:
    """Most positive/negative words according to the linear model's coefficients."""
    clf = pipe.named_steps["clf"]
    word_vec = pipe.named_steps["features"].transformer_list[0][1]
    n_word = len(word_vec.vocabulary_)
    vocab = np.array(word_vec.get_feature_names_out())
    classes = list(clf.classes_)
    out = {}
    for ci, cls in enumerate(classes):
        coefs = clf.coef_[ci][:n_word]
        order = np.argsort(coefs)[::-1][:k]
        out[cls] = [(vocab[i], float(coefs[i])) for i in order]
    return out


def plot_top_terms(terms: dict) -> None:
    viz.setup()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    hidden = {"bc", "mc"}  # abusive abbreviations: real model features, but not shown on slides
    for ax, cls in zip(axes, LABELS):
        words, weights = zip(*[(w, c) for w, c in terms[cls] if w not in hidden][:12])
        y = np.arange(len(words))[::-1]
        ax.barh(y, weights, color=viz.SENTIMENT_COLORS[cls], height=0.62)
        ax.set_yticks(y, words)
        ax.set_title(cls, color=viz.INK, fontsize=12)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("model weight")
    fig.suptitle("Words the model leans on most, per sentiment class", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    viz.finish(fig, "03_top_words_per_class",
               "Neutral is a catch-all class (tags, news-style statements), so its top words are less meaningful. "
               "Profanity abbreviations are hidden from this display.")


def main() -> None:
    ensure_dirs()
    df, _ = load_and_clean()
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE
    )
    print(f"train={len(X_train)}  test={len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}
    for name, clf in candidate_models().items():
        s = cross_val_score(make_pipeline(clf), X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        cv_scores[name] = (float(s.mean()), float(s.std()))
        print(f"  {name:28s} macro-F1 = {s.mean():.4f} ± {s.std():.4f}")
    plot_model_comparison(cv_scores)

    # Tune the strongest linear model with a small, honest grid (still only on the training folds).
    best_name = max((n for n in cv_scores if n != "Majority-class baseline"), key=lambda n: cv_scores[n][0])
    print(f"Best by CV: {best_name}")
    if best_name == "Linear SVM":
        grid = {"C": [0.03, 0.05, 0.1, 0.2, 0.4]}
        make = lambda c: LinearSVC(C=c, class_weight="balanced")
    elif best_name == "Logistic Regression":
        grid = {"C": [1, 2, 5, 10, 20]}
        make = lambda c: LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
    else:
        grid = {"C": [0.1, 0.3, 0.6, 1.0]}
        make = lambda c: ComplementNB(alpha=c)
    tuned = {}
    for c in grid["C"]:
        s = cross_val_score(make_pipeline(make(c)), X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        tuned[c] = float(s.mean())
        print(f"    param={c:<5} CV macro-F1 = {s.mean():.4f}")
    best_param = max(tuned, key=tuned.get)

    final = make_pipeline(make(best_param)).fit(X_train, y_train)
    pred = final.predict(X_test)

    report = classification_report(y_test, pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=LABELS)
    base = make_pipeline(DummyClassifier(strategy="most_frequent")).fit(X_train, y_train)
    metrics = {
        "model": best_name,
        "best_param": best_param,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "cv_macro_f1": cv_scores,
        "tuning_cv_macro_f1": {str(k): v for k, v in tuned.items()},
        "test_accuracy": float((pred == y_test).mean()),
        "test_macro_f1": float(f1_score(y_test, pred, average="macro")),
        "baseline_test_macro_f1": float(f1_score(y_test, base.predict(X_test), average="macro")),
        "per_class": {k: {m: float(v) for m, v in report[k].items()} for k in LABELS},
        "confusion_matrix": cm.tolist(),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(cm)

    if hasattr(final.named_steps["clf"], "coef_"):
        terms = top_terms(final)
        plot_top_terms(terms)
        (REPORTS_DIR / "top_terms.json").write_text(json.dumps(terms, indent=2))

    # Error analysis: keep real mistakes so we can discuss them honestly.
    errs = pd.DataFrame({"text": X_test.values, "actual": y_test.values, "predicted": pred})
    errs[errs.actual != errs.predicted].head(60).to_csv(REPORTS_DIR / "misclassified_examples.csv", index=False)

    joblib.dump(final, MODEL_PATH)
    print(f"\nTEST accuracy={metrics['test_accuracy']:.4f}  macro-F1={metrics['test_macro_f1']:.4f}  "
          f"(baseline macro-F1={metrics['baseline_test_macro_f1']:.4f})")
    print(classification_report(y_test, pred, labels=LABELS, zero_division=0))


if __name__ == "__main__":
    main()
