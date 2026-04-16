#!/usr/bin/env python

from pathlib import Path
import sys
import io

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
REPORT_PATH = Path("data/raw/data_ingestion_report.txt")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Data loaded successfully with shape: {df.shape}")
    return df


def basic_info(df: pd.DataFrame) -> None:
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_text = buffer.getvalue()
    describe_text = df.describe(include="all").to_string()
    missing_text = df.isnull().sum().to_string()

    report = (
        "Basic Information about the dataset:\n"
        f"{info_text}\n"
        "Summary Statistics:\n"
        f"{describe_text}\n\n"
        "Missing Values:\n"
        f"{missing_text}\n"
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Basic Information about the dataset:")
    print(info_text)
    print("Summary Statistics:")
    print(describe_text)
    print("\nMissing Values:")
    print(missing_text)
    print(f"\nIngestion report saved to: {REPORT_PATH}")


def main() -> int:
    df = load_data()
    basic_info(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
