from pathlib import Path
import sys

try:
    import pandas as pd
    from dotenv import load_dotenv
except ModuleNotFoundError as exc:
    missing_package = exc.name or "a required package"
    raise SystemExit(
        f"Missing dependency: {missing_package}. "
        "Activate the project virtualenv with `source .venv/bin/activate` "
        "or run the script with `.venv/bin/python src/data_ingestion.py`."
    ) from exc


load_dotenv()

RAW_DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Data loaded successfully with shape: {df.shape}")
    return df


def basic_info(df: pd.DataFrame) -> None:
    print("Basic Information about the dataset:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())


def main() -> int:
    df = load_data()
    basic_info(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
