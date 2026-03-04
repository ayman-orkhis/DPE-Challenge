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
    reference_dir = Path(reference_dir)
    prediction_dir = Path(prediction_dir)
    output_dir = Path(output_dir)
    
    scores = {}
    for eval_set in EVAL_SETS:
        print(f'Scoring {eval_set}')
        
        # Check if prediction file exists
        pred_file = prediction_dir / f'{eval_set}_predictions.csv'
        if not pred_file.exists():
            print(f'  Warning: Prediction file not found at {pred_file}')
            print(f'  Available files in {prediction_dir}: {list(prediction_dir.glob("*"))}')
            continue
        
        # Check if reference file exists
        ref_file = reference_dir / f'{eval_set}_labels.csv'
        if not ref_file.exists():
            print(f'  Warning: Reference file not found at {ref_file}')
            print(f'  Available files in {reference_dir}: {list(reference_dir.glob("*"))}')
            continue

        predictions = pd.read_csv(pred_file)
        targets = pd.read_csv(ref_file)

        rmse = compute_rmse(predictions, targets)
        mae = compute_mae(predictions, targets)

        scores[eval_set] = rmse
        scores[f'{eval_set}_mae'] = mae

        # print(f'  RMSE: {rmse:.4f}')
        # print(f'  MAE:  {mae:.4f}')

    # Add train and test times in the score
    metadata_file = prediction_dir / 'metadata.json'
    if metadata_file.exists():
        json_durations = metadata_file.read_text()
        durations = json.loads(json_durations)
        scores.update(**durations)
    
    # print(scores)

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
