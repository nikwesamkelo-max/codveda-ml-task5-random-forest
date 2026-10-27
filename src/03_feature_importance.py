"""Step 3 - Feature importance analysis: what drives the churn predictions?"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "artifacts/random_forest.joblib"
TABLE_PATH = "outputs/feature_importance.csv"
CHART_PATH = "images/feature_importance.png"
TARGET = "Churn"
RANDOM_STATE = 42


def load_xy(path):
    df = pd.read_csv(path)
    return df.drop(columns=TARGET), df[TARGET]


def impurity_importance(model, features):
    """Built-in importance: how much each feature reduces impurity across all
    trees (averaged). Fast, but can favour continuous features."""
    per_tree = np.array([t.feature_importances_ for t in model.estimators_])
    return pd.DataFrame({"impurity": model.feature_importances_,
                         "impurity_std": per_tree.std(axis=0)}, index=features)


def permutation_importance_table(model, X, y, features):
    """Shuffle one feature at a time on UNSEEN data and measure how much the
    F1-score drops. A big drop means the model really relies on that feature."""
    result = permutation_importance(model, X, y, scoring="f1", n_repeats=30,
                                    random_state=RANDOM_STATE, n_jobs=-1)
    return pd.DataFrame({"permutation": result.importances_mean,
                         "permutation_std": result.importances_std}, index=features)


def direction_table(X, y):
    """Average of each feature for churners vs customers who stayed."""
    means = X.groupby(y).mean().T
    means.columns = ["stayed_avg", "churned_avg"]
    means["churned_vs_stayed_%"] = (means["churned_avg"] / means["stayed_avg"].replace(0, np.nan) - 1) * 100
    return means


def plot_importance(table):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, col, color, title, xlabel in (
            (axes[0], "impurity", "#2563eb", "Built-in importance", "Share of the model's decisions"),
            (axes[1], "permutation", "#f97316", "Permutation importance (unseen data)",
             "Drop in F1-score when the feature is shuffled")):
        t = table.sort_values(col)
        ax.barh(t.index, t[col], xerr=t[f"{col}_std"], color=color, capsize=3)
        ax.set_title(title, fontsize=16, weight="bold")
        ax.set_xlabel(xlabel)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.subplots_adjust(wspace=0.65)  # room for the right-hand labels
    fig.suptitle("What drives the churn predictions?", fontsize=20, weight="bold")
    os.makedirs("images", exist_ok=True)
    fig.savefig(CHART_PATH, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    model = joblib.load(MODEL_PATH)
    X_train, y_train = load_xy(TRAIN_PATH)
    X_test, y_test = load_xy(TEST_PATH)
    features = list(X_train.columns)

    table = impurity_importance(model, features).join(
        permutation_importance_table(model, X_test, y_test, features))
    table = table.join(direction_table(X_train, y_train)[["churned_vs_stayed_%"]])
    table = table.sort_values("permutation", ascending=False)

    show = table[["impurity", "permutation", "churned_vs_stayed_%"]].round(3)
    print("Feature importance (ranked by permutation importance):\n")
    print(show.to_string())
    top_imp = set(table["impurity"].nlargest(5).index)
    top_perm = set(table["permutation"].nlargest(5).index)
    print(f"\nTop 5 by both methods: {sorted(top_imp & top_perm)}")

    # Checks
    assert abs(table["impurity"].sum() - 1) < 1e-9, "Importances should sum to 1"
    assert set(table.index) == set(features)
    assert table["permutation"].iloc[0] > 0, "Top feature should matter"
    print("Feature importance verified")

    os.makedirs("outputs", exist_ok=True)
    table.round(4).to_csv(TABLE_PATH)
    plot_importance(table)
    print(f"Saved -> {TABLE_PATH}, {CHART_PATH}")


if __name__ == "__main__":
    main()
