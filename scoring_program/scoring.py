import json
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_SETS = ["test", "private_test"]


def compute_rmse(predictions, targets):
    """Compute Root Mean Squared Error."""
    predictions = predictions.values.ravel().astype(float)
    targets = targets.values.ravel().astype(float)
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def compute_mae(predictions, targets):
    """Compute Mean Absolute Error."""
    predictions = predictions.values.ravel().astype(float)
    targets = targets.values.ravel().astype(float)
    return float(np.mean(np.abs(predictions - targets)))


def main(reference_dir, prediction_dir, output_dir):
    scores = {}
    for eval_set in EVAL_SETS:
        print(f'Scoring {eval_set}')

        predictions = pd.read_csv(
            prediction_dir / f'{eval_set}_predictions.csv'
        )
        targets = pd.read_csv(
            reference_dir / f'{eval_set}_labels.csv'
        )

        rmse = compute_rmse(predictions, targets)
        mae = compute_mae(predictions, targets)

        scores[eval_set] = rmse
        scores[f'{eval_set}_mae'] = mae

        print(f'  RMSE: {rmse:.4f}')
        print(f'  MAE:  {mae:.4f}')

    # Add train and test times in the score
    json_durations = (prediction_dir / 'metadata.json').read_text()
    durations = json.loads(json_durations)
    scores.update(**durations)
    print(scores)

    # Write output scores
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'scores.json').write_text(json.dumps(scores))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scoring program for codabench"
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="/app/input/ref",
        help="",
    )
    parser.add_argument(
        "--prediction-dir",
        type=str,
        default="/app/input/res",
        help="",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/output",
        help="",
    )

    args = parser.parse_args()

    main(
        Path(args.reference_dir),
        Path(args.prediction_dir),
        Path(args.output_dir)
    )
