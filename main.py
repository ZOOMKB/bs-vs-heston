from pathlib import Path

import pandas as pd

from src.market_data import ENRICHED_OPTIONS_PATH, print_summary


def main() -> None:
    if not Path(ENRICHED_OPTIONS_PATH).exists():
        raise FileNotFoundError("Run notebook/01_data_acquisition.ipynb before using the summary entrypoint.")

    options = pd.read_csv(ENRICHED_OPTIONS_PATH)
    print_summary(options)


if __name__ == "__main__":
    main()
