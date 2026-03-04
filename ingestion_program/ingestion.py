import json
import sys
import time
import zipfile
from pathlib import Path
import subprocess
import sys
import importlib


import pandas as pd

try:
    from bench_utils import TARGET_COLUMN, LEAK_COLUMNS
except ModuleNotFoundError:
    try:
        from .bench_utils import TARGET_COLUMN, LEAK_COLUMNS
    except (ModuleNotFoundError, ImportError):
        # Fallback: add the ingestion_program to path
        sys.path.insert(0, str(Path(__file__).parent))
        from bench_utils import TARGET_COLUMN, LEAK_COLUMNS


EVAL_SETS = ["test", "private_test"]


def evaluate_model(model, X_test):
    y_pred = model.predict(X_test)
    return pd.DataFrame(y_pred, columns=["prediction"])


def clean_features(df):
    """Remove target and leakage columns from the feature dataframe."""
    cols_to_drop = [TARGET_COLUMN] + LEAK_COLUMNS
    cols_present = [c for c in cols_to_drop if c in df.columns]
    if cols_present:
        print(f"  Dropping columns: {cols_present}")
    return df.drop(columns=cols_present, errors="ignore")


def get_train_data(data_dir):
    data_dir = Path(data_dir)
    training_dir = data_dir / "train"
    X_train = pd.read_csv(training_dir / "train_features.csv")
    y_train = pd.read_csv(training_dir / "train_labels.csv")
    X_train = clean_features(X_train)
    return X_train, y_train



def main(data_dir, output_dir, submission_dir):
    # Import the submission module
    # Add the solution directory to the path``

    sys.path.insert(0, str(submission_dir))
    
    from submission import get_model




    X_train, y_train = get_train_data(data_dir)

    print("Training the model")
    print(f"  Features shape: {X_train.shape}")
    print(f"  Target shape: {y_train.shape}")

    model = get_model()

    start = time.time()
    model.fit(X_train, y_train.values.ravel())
    train_time = time.time() - start
    print(f"  Training time: {train_time:.2f}s")
    print("-" * 40)

    print("Evaluate the model")
    start = time.time()
    res = {}
    for eval_set in EVAL_SETS:
        X_test = pd.read_csv(
            data_dir / eval_set / f"{eval_set}_features.csv"
        )
        X_test = clean_features(X_test)
        res[eval_set] = evaluate_model(model, X_test)
        print(f"  {eval_set}: {len(res[eval_set])} predictions")
    test_time = time.time() - start
    print("-" * 40)

    duration = train_time + test_time
    print(f"Completed Prediction. Total duration: {duration:.2f}s")

    # Write output files
    # Create solution zip file
    # Write output files
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metadata.json", "w+") as f:
        json.dump(dict(train_time=train_time, test_time=test_time), f)
    for eval_set in EVAL_SETS:
        filepath = output_dir / f"{eval_set}_predictions.csv"
        res[eval_set].to_csv(filepath, index=False)
        print(f"Predictions for {eval_set} written to {filepath}")

    # Create solution zip file
    solution_dir = Path(__file__).parent.parent / "solution"
    submission_file = solution_dir / "submission.py"
    
    if submission_file.exists():
        solution_zip_path = output_dir / "solution.zip"
        print(f"Creating solution zip file: {solution_zip_path}")
        with zipfile.ZipFile(solution_zip_path, 'w') as zipf:
            zipf.write(submission_file, arcname="submission.py")
        print(f"Solution zip file created successfully at {solution_zip_path}")
    else:
        print(f"Warning: submission.py not found at {submission_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingestion program for codabench"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/app/input_data",
        help="",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/output",
        help="",
    )
    parser.add_argument(
        "--submission-dir",
        type=str,
        default="/app/ingested_program",
        help="",
    )

    args = parser.parse_args()
    
    # Add required paths
    ingestion_dir = Path(__file__).parent.resolve()
    if str(ingestion_dir) not in sys.path:
        sys.path.append(str(ingestion_dir))
    
    if args.submission_dir and Path(args.submission_dir).exists():
        if str(args.submission_dir) not in sys.path:
            sys.path.append(str(args.submission_dir))
    
    main(Path(args.data_dir), Path(args.output_dir), Path(args.submission_dir))
