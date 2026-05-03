# SPY Options Smile: Black-Scholes and Heston

This project downloads SPY option chains, enriches them with Treasury rates and dividend yield estimates, and compares Black-Scholes implied volatility with a calibrated Heston stochastic volatility model.

## Project Structure

- `notebook/01_data_acquisition.ipynb` downloads raw option chains and creates the enriched dataset.
- `notebook/02_black_scholes.ipynb` implements Black-Scholes checks, put-call parity, implied volatility, and volatility-smile plots.
- `notebook/03_heston_calibration.ipynb` validates Heston pricing, prepares the filtered calibration dataset, and runs the Heston optimizer.
- `notebook/04_comparison.ipynb` overlays the calibrated Heston smile on market IV and compares Delta/Vega sensitivities.
- `src/` contains reusable data, pricing, calibration, Greeks, and plotting code.
- `test/` contains the pytest suite.

## Setup

```bash
uv sync
```

## Running the Analysis

Run the notebooks in order:

```text
notebook/01_data_acquisition.ipynb
notebook/02_black_scholes.ipynb
notebook/03_heston_calibration.ipynb
notebook/04_comparison.ipynb
```

The Heston calibration in `03_heston_calibration.ipynb` can take several minutes. Run it when you want to recalibrate the model. After calibration, copy the resulting parameters into `04_comparison.ipynb` so the comparison plots and final analysis use the latest fitted model.

## Tests

```bash
uv run pytest
```

## Data

The current data files are:

- `data/spy_options_raw.csv`
- `data/spy_options_enriched.csv`

Re-running the data acquisition notebook will refresh both files.
