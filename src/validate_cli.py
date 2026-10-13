"""Data validation entry point for Airflow/Jenkins: exits non-zero on bad data.

    python -m src.validate_cli
"""
import sys

from src.data import validate

if __name__ == "__main__":
    try:
        print("data ok:", validate())
    except (ValueError, FileNotFoundError) as e:
        print(f"data validation failed: {e}", file=sys.stderr)
        sys.exit(1)
