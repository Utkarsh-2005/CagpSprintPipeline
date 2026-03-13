# Car Manufacturing Project Architecture

## Overview
This pipeline converts a messy car-rental style operational dataset into a cleaned dataset, validates core data quality constraints, and runs 20 analytics transformations with chart exports.

## Layers
- Ingestion layer: reads data from `data/messy_dataset.csv`.
- Cleaning layer (`src/cleaning/cleaner.py`): applies normalization, parsing, masking, and rule-based fixes.
- Validation layer (`src/validation/validator.py`): computes data quality checks and writes a report.
- Analytics layer (`src/analytics/transformer.py`): computes 20 transformation scenarios and saves visual outputs.
- Orchestration layer (`src/pipeline/pipeline_runner.py`): executes all steps and returns run summary.

## Data Flow
1. `main.py` calls `src/pipeline/run_pipeline.py`.
2. Runner loads messy dataset.
3. Cleaner standardizes IDs, timestamps, odometer, fuel, rate, city, payment, GPS, speed, promo logic, and totals.
4. Cleaned output is stored in `data/cleaned_dataset.csv`.
5. Validator writes `docs/validation_report.csv`.
6. Transformer generates scenario tables in memory and charts in `visualizations/charts`.

## Outputs
- Cleaned dataset: `data/cleaned_dataset.csv`
- Validation report: `docs/validation_report.csv`
- Charts: `visualizations/charts/*.png`
