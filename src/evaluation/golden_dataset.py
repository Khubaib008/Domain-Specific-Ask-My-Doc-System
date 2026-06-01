"""Golden dataset loader for RAG evaluation."""

import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "golden_dataset.json"


def load_golden_dataset(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Load the golden Q&A dataset from JSON.

    Returns a DataFrame with columns: id, question, reference_answer.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    pairs = data["pairs"]
    df = pd.DataFrame(pairs)
    print(f"Loaded {len(df)} Q&A pairs from {path}")
    return df


def save_golden_dataset(df: pd.DataFrame, path: str | Path = DEFAULT_PATH) -> None:
    """Save a golden dataset DataFrame back to JSON."""
    path = Path(path)
    pairs = df.to_dict(orient="records")
    data = {
        "description": "Golden dataset for RAG evaluation",
        "pairs": pairs,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(pairs)} Q&A pairs to {path}")
