# Script to prepare competition data from curated DPE parquet files.
# Expects train.parquet and test.parquet in the project root.
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add parent so we can import bench_utils
sys.path.append(str(Path(__file__).parent.parent / "ingestion_program"))
from bench_utils import TARGET_COLUMN, LEAK_COLUMNS

PHASE = 'dev_phase'
DATA_DIR = Path(PHASE) / 'input_data'
REF_DIR = Path(PHASE) / 'reference_data'

ROOT_DIR = Path(__file__).parent.parent


def make_parquet(df, filepath):
    """Save a dataframe as a parquet file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False)


def make_csv(df, filepath):
    """Save a dataframe as a CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='Prepare competition data from curated DPE parquet files'
    )
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/test split')
    parser.add_argument('--train-file', type=str,
                        default=str(ROOT_DIR / 'train.parquet'),
                        help='Path to curated train.parquet')
    parser.add_argument('--test-file', type=str,
                        default=str(ROOT_DIR / 'test.parquet'),
                        help='Path to curated test.parquet')
    args = parser.parse_args()

    # --- Load curated data ---
    print(f"Loading training data from: {args.train_file}")
    train_df = pd.read_parquet(args.train_file)
    print(f"  Shape: {train_df.shape}")

    print(f"Loading test data from: {args.test_file}")
    test_df = pd.read_parquet(args.test_file)
    print(f"  Shape: {test_df.shape}")

    # --- Extract target ---
    assert TARGET_COLUMN in train_df.columns, (
        f"Target column '{TARGET_COLUMN}' not found in train data. "
        f"Available columns: {list(train_df.columns)}"
    )
    assert TARGET_COLUMN in test_df.columns, (
        f"Target column '{TARGET_COLUMN}' not found in test data. "
        f"Available columns: {list(test_df.columns)}"
    )

    y_train = train_df[[TARGET_COLUMN]]
    y_test_full = test_df[[TARGET_COLUMN]]

    # Features: keep all columns (leakage columns will be dropped at
    # ingestion time, but we also drop them here for the feature files)
    cols_to_drop = [TARGET_COLUMN] + [
        c for c in LEAK_COLUMNS if c in train_df.columns
    ]
    X_train = train_df.drop(columns=cols_to_drop)
    X_test_full = test_df.drop(columns=[
        c for c in cols_to_drop if c in test_df.columns
    ])

    print(f"Features after dropping target + leakage: {X_train.shape[1]} cols")
    print(f"Leaked columns dropped: {[c for c in LEAK_COLUMNS if c in train_df.columns]}")

    # --- Split test into public and private test ---
    rng = np.random.RandomState(args.seed)
    X_test, X_private_test, y_test, y_private_test = train_test_split(
        X_test_full, y_test_full, test_size=0.5, random_state=rng
    )
    print(f"Public test: {X_test.shape[0]} samples")
    print(f"Private test: {X_private_test.shape[0]} samples")

    # --- Save to dev_phase structure ---
    # input_data/train/ : features (parquet) + labels (csv)
    # input_data/test/ : features only (parquet)
    # input_data/private_test/ : features only (parquet)
    # reference_data/ : test and private_test labels (csv)

    for split, X_split, y_split in [
        ('train', X_train, y_train),
        ('test', X_test, y_test),
        ('private_test', X_private_test, y_private_test),
    ]:
        split_dir = DATA_DIR / split
        make_parquet(X_split, split_dir / f'{split}_features.parquet')

        # Train labels go in input_data, test labels in reference_data
        label_dir = split_dir if split == "train" else REF_DIR
        make_csv(y_split, label_dir / f'{split}_labels.csv')

    print()
    print(f"Data setup complete. Files written to '{PHASE}/'")