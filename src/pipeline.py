"""Run the full training pipeline in one go (used by `make train`, Jenkins and Docker).

    python -m src.pipeline
"""
import subprocess
import sys

from src import train_flight_price, train_gender, train_recommender
from src.data import validate


def main():
    print("data:", validate())
    subprocess.run([sys.executable, "-m", "src.monitor"], check=True, stdout=subprocess.DEVNULL)
    train_flight_price.main()
    train_gender.main()
    train_recommender.main()


if __name__ == "__main__":
    main()
