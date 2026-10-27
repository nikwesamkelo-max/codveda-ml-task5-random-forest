"""Step 2 - Evaluate the Random Forest with cross-validation and classification metrics."""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedKFold, cross_validate

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "artifacts/random_forest.joblib"
METRICS_PATH = "outputs/metrics.json"
REPORT_PATH = "outputs/classification_report.txt"
CHART_PATH = "images/evaluation.png"
TARGET = "Churn"
RANDOM_STATE = 42
SCORING = ["accuracy", "precision", "recall", "f1"]


def load_xy(path):
    df = pd.read_csv(path)
    return df.drop(columns=TARGET), df[TARGET]


def cross_validate_model(model, X, y):
    """5-fold cross-validation on the training data: train on 4/5, score on the
    held-out 1/5, rotate. Gives a mean score and how much it varies."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(clone(model), X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    return {m: (scores[f"test_{m}"].mean(), scores[f"test_{m}"].std()) for m in SCORING}


def test_metrics(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return pred, {"accuracy": accuracy_score(y, pred),
                  "precision": precision_score(y, pred),
                  "recall": recall_score(y, pred),
                  "f1": f1_score(y, pred),
                  "roc_auc": roc_auc_score(y, proba)}


def plot_evaluation(y_true, y_pred, cv, test):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred),
                           display_labels=["Stayed", "Churned"]).plot(
        cmap="Blues", colorbar=False, ax=axes[0], values_format="d")
    axes[0].set_title("Predictions on 667 unseen customers", fontsize=16, weight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    names = ["precision", "recall", "f1"]
    x = np.arange(len(names)); w = 0.38
    cv_vals = [cv[n][0] for n in names]; cv_err = [cv[n][1] for n in names]
    test_vals = [test[n] for n in names]
    b1 = axes[1].bar(x - w / 2, cv_vals, w, yerr=cv_err, capsize=5, color="#2563eb",
                     label="Cross-validation (training)")
    b2 = axes[1].bar(x + w / 2, test_vals, w, color="#f97316", label="Unseen test customers")
    axes[1].bar_label(b1, fmt="%.2f", padding=3, fontsize=12)
    axes[1].bar_label(b2, fmt="%.2f", padding=3, fontsize=12)
    axes[1].set_xticks(x); axes[1].set_xticklabels([n.capitalize() if n != "f1" else "F1-score" for n in names])
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Churn class metrics", fontsize=16, weight="bold")
    axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    for side in ("top", "right"):
        axes[1].spines[side].set_visible(False)
    os.makedirs("images", exist_ok=True)
    fig.savefig(CHART_PATH, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    model = joblib.load(MODEL_PATH)
    X_train, y_train = load_xy(TRAIN_PATH)
    X_test, y_test = load_xy(TEST_PATH)

    cv = cross_validate_model(model, X_train, y_train)
    print("Cross-validation on training data (mean +/- std over 5 folds):")
    for m, (mean, std) in cv.items():
        print(f"  {m:<10}{mean:.3f} +/- {std:.3f}")

    pred, test = test_metrics(model, X_test, y_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    baseline = 1 - y_test.mean()
    print(f"\nUnseen test customers ({len(y_test)}):")
    for m, v in test.items():
        print(f"  {m:<10}{v:.3f}")
    print(f"  (always guessing 'no churn' would give accuracy {baseline:.3f} and recall 0)")
    print(f"\nOf {tp + fn} real churners: caught {tp}, missed {fn}")
    print(f"Of {tp + fp} customers flagged as churn: {tp} really churned, {fp} were false alarms")

    report = classification_report(y_test, pred, target_names=["Stayed", "Churned"], digits=3)
    print("\n" + report)

    # Checks
    assert len(pred) == len(y_test)
    assert tn + fp + fn + tp == len(y_test)
    assert abs(test["f1"] - 2 * test["precision"] * test["recall"] / (test["precision"] + test["recall"])) < 1e-9
    assert test["recall"] > 0, "Model should catch some churners"
    print("Evaluation verified")

    os.makedirs("outputs", exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump({"cross_validation": {m: {"mean": a, "std": s} for m, (a, s) in cv.items()},
                   "test": test, "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                                                      "fn": int(fn), "tp": int(tp)}}, f, indent=2)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    plot_evaluation(y_test, pred, cv, test)
    print(f"Saved -> {METRICS_PATH}, {REPORT_PATH}, {CHART_PATH}")


if __name__ == "__main__":
    main()
