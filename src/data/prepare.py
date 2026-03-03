# src/prepare.py

from pathlib import Path
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

# Local imports
from src.constants import DATA_RAW_DIR, DATA_PROCESSED_DIR


RANDOM_STATE = 42
TEST_SIZE = 0.2


def main(input_file: Path, output_dir: Path, test_size: float, seed: int):
    # Checking input material
    assert input_file.is_file(), f"Input file {input_file} does not exist."

    # Create output dir if not exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(input_file.absolute())

    # Suppress non relevant data
    df = df.drop(columns=['date'])

    # Target column
    target_col = "silica_concentrate"

    # Split targets and features
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed
    )

    # Save outputs
    X_train.to_csv(output_dir/"X_train.csv", index=False)
    X_test.to_csv(output_dir/"X_test.csv", index=False)
    y_train.to_csv(output_dir/"y_train.csv", index=False)
    y_test.to_csv(output_dir/"y_test.csv", index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Prepare dataset for training and testing.")
    args.add_argument(
        "--input_path",
        type=Path,
        default=DATA_RAW_DIR / "raw.csv",
        help="Path to the raw dataset.",
    )
    args.add_argument(
        "--output_dir",
        type=Path,
        default=DATA_PROCESSED_DIR,
        help="Directory to save the processed dataset.",
    )
    args.add_argument(
        "--test_size",
        type=float,
        default=TEST_SIZE,
        help="Size of the test sample.",
    )
    args.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for train/test split.",
    )
    # Parsing arguments
    args = args.parse_args()

    # Run the main process
    main(
        input_file=args.input_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
    )
