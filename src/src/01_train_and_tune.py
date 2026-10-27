"""Step 1 - Train a Random Forest and tune its hyperparameters."""
import os
import json
import time
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

TRAIN_RAW = "data/raw/churn-bigml-80.csv"
TEST_RAW = "data/raw/churn-bigml-20.csv"
OUT_DIR = "data/processed"
MODEL_PATH = "artifacts/random_forest.joblib"
PARAMS_PATH = "outputs/best_params.json"
GRID_PATH = "outputs/tuning_results.csv"
CHART_PATH = "images/tuning_heatmap.png"
TARGET = "Churn"
RANDOM_STATE = 42

# Dropped columns: charges duplicate the minutes columns; State and Area code
# are identifiers (adding State as 51 extra columns made the model worse).
DROP_COLS = ["Total day charge", "Total eve charge", "Total night charge",
             "Total intl charge", "Area code", "State"]
PARAM_GRID = {"n_estimators": [50, 100, 200, 300, 500],
              "max_depth": [3, 5, 8, 10, 15, None]}


def preprocess(path):
    """Load a churn file, drop unhelpful columns, turn Yes/No into 1/0."""
    df = pd.read_csv(path).drop(columns=DROP_COLS)
    for col in ("International plan", "Voice mail plan"):
        df[col] = (df[col] == "Yes").astype(int)
    df[TARGET] = df[TARGET].astype(int)
    return df.drop(columns=TARGET), df[TARGET]


def tune(X, y, cv):
    """Try every combination of trees x depth, scored by cross-validated F1."""
    search = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE),
                          PARAM_GRID, scoring="f1", cv=cv, n_jobs=-1)
    return search.fit(X, y)


def plot_heatmap(results):
    table = results.pivot_table(index="max_depth", columns="n_estimators", values="f1")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.heatmap(table, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=1,
                cbar_kws={"label": "Cross-validated F1-score"}, ax=ax)
    ax.set_title("Tuning the Random Forest: trees vs depth", fontsize=19, weight="bold", loc="left")
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Max depth")
    os.makedirs("images", exist_ok=True)
    fig.savefig(CHART_PATH, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    X_train, y_train = preprocess(TRAIN_RAW)
    X_test, y_test = preprocess(TEST_RAW)
    print(f"Train: {len(X_train)} customers | Test: {len(X_test)} customers | Features: {X_train.shape[1]}")
    print(f"Churn rate: {y_train.mean():.1%} (always guessing 'no churn' would be "
          f"{1 - y_train.mean():.1%} accurate, so accuracy alone is misleading)\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    default_f1 = cross_val_score(RandomForestClassifier(random_state=RANDOM_STATE),
                                 X_train, y_train, cv=cv, scoring="f1").mean()

    start = time.time()
    search = tune(X_train, y_train, cv)
    n_fits = len(search.cv_results_["params"]) * cv.get_n_splits()
    print(f"Tried {len(search.cv_results_['params'])} combinations ({n_fits} fits) in {time.time() - start:.0f}s")

    results = pd.DataFrame(search.cv_results_["params"]).assign(
        f1=search.cv_results_["mean_test_score"], f1_std=search.cv_results_["std_test_score"])
    results["max_depth"] = results["max_depth"].fillna(-1).astype(int).replace(-1, "None")
    print("\nTop 5 settings:")
    print(results.sort_values("f1", ascending=False).head(5).round(3).to_string(index=False))

    best = search.best_params_
    print(f"\nBest: {best['n_estimators']} trees, max depth {best['max_depth']}")
    print(f"Cross-validated F1: tuned {search.best_score_:.3f} vs default settings {default_f1:.3f}")

    # Checks
    model = search.best_estimator_
    assert model.n_features_in_ == X_train.shape[1]
    assert search.best_score_ >= default_f1 - 1e-9, "Tuned model should be at least as good as default"
    assert X_train.isna().sum().sum() == 0 and X_test.isna().sum().sum() == 0
    assert list(X_train.columns) == list(X_test.columns)
    print("\nTuning verified. Test data untouched until step 2")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    X_train.assign(**{TARGET: y_train}).to_csv(f"{OUT_DIR}/train.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(f"{OUT_DIR}/test.csv", index=False)
    joblib.dump(model, MODEL_PATH)
    with open(PARAMS_PATH, "w") as f:
        json.dump({"best_params": best, "cv_f1_tuned": search.best_score_,
                   "cv_f1_default": default_f1}, f, indent=2)
    results.round(4).to_csv(GRID_PATH, index=False)
    plot_heatmap(results)
    print(f"Saved -> {MODEL_PATH}, {PARAMS_PATH}, {GRID_PATH}, {CHART_PATH}")


if __name__ == "__main__":
    main()
