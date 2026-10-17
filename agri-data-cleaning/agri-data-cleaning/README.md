# Agribusiness Data Cleaning and Preprocessing Pipeline

Companion project for the **Week 2 task: Data Cleaning and Preprocessing Strategy** (Junior Data Scientist Internship, Nagpur).

**Author:** Sumit Kumar Sharma

This repository turns the strategy described in the report into a small, reproducible pipeline. It uses a **synthetic (hypothetical)** dataset, *AgriFarm Survey 2025-26*, that imitates farm survey data from the Vidarbha region of Maharashtra and contains realistic quality problems.

## Problems injected into the raw data

| Problem | Example |
|---|---|
| Missing values | about 18% of soil pH values blank |
| Placeholder codes | rainfall = -999, fertiliser = "NA" |
| Inconsistent names | `Soybean`, `SOYA BEAN `, `soyabean`, `Kapas` for cotton |
| Mixed units | area in hectare, acre and bigha |
| Mixed date formats | `12/07/25`, `2025-07-12`, `12 Jul 2025` |
| Impossible values | pH = 15, area = -1 |
| Outliers | yields 10x too high (decimal-point errors) |
| Duplicates | 12 repeated rows |

## Pipeline (clean_pipeline.py)

1. Read raw data as text (raw file is never modified)
2. Replace placeholders with true missing values and fix data types
3. Standardise crop/district names, dates (ISO 8601) and area (hectare)
4. Remove duplicates
5. Domain-rule range checks and crop-specific yield limits (impossible values become missing)
6. Group-wise IQR outlier flagging (genuine extremes are kept and flagged)
7. Missing-value imputation with district-wise median, plus `*_was_missing` indicator columns; rows without target (yield) are dropped, never imputed
8. Logical consistency check
9. Feature engineering, one-hot encoding, standardisation with parameters learned **on the training split only** (no data leakage)
10. Write cleaned data, model-ready data, a cleaning log and a before/after quality report

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python generate_data.py          # creates data/raw/agrifarm_raw.csv (seed = 42)
python clean_pipeline.py         # creates files in data/processed and reports
```

## Outputs

| File | Description |
|---|---|
| `data/raw/agrifarm_raw.csv` | messy synthetic input |
| `data/processed/agrifarm_clean.csv` | cleaned, human-readable data with flags |
| `data/processed/agrifarm_model_ready.csv` | encoded and scaled features with train/test split column |
| `reports/cleaning_log.csv` | records affected by each cleaning rule |
| `reports/quality_before_after.csv` | duplicates and missing percentage before vs after |

## Reproducibility

- Fixed random seed (42) for data generation and the train/test split
- All thresholds, unit factors and name mappings are in the configuration block at the top of `clean_pipeline.py`
- Every rule writes to the cleaning log, so each run is auditable
- Running the two scripts again gives identical outputs

## Note

All data and conversion factors (for example the bigha-to-hectare factor) are hypothetical and for demonstration only. They must not be used for real agronomic decisions.
