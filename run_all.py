"""Run the full Random Forest pipeline (steps 1-3) in order."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_FILES = [ROOT / "data" / "raw" / "churn-bigml-80.csv",
             ROOT / "data" / "raw" / "churn-bigml-20.csv"]
STEPS = [
    "src/01_train_and_tune.py",
    "src/02_evaluate_model.py",
    "src/03_feature_importance.py",
]


def main():
    missing = [str(f) for f in RAW_FILES if not f.exists()]
    if missing:
        sys.exit("Dataset not found:\n  " + "\n  ".join(missing) +
                 "\nPlace both churn CSV files in data/raw/ first.")

    print("Note: step 1 tunes 30 settings with cross-validation and takes a few minutes.")
    total_start = time.time()
    for number, step in enumerate(STEPS, start=1):
        print(f"\n{'=' * 60}\nSTEP {number}/{len(STEPS)}: {step}\n{'=' * 60}")
        start = time.time()
        result = subprocess.run([sys.executable, step], cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f"\nStep failed: {step}")
        print(f"Finished in {time.time() - start:.1f}s")

    print(f"\nPipeline complete in {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
