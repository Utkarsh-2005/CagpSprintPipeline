# Car Manufacturing Project

## Run
From the project root:

```bash
python main.py
```

## Pipeline Modules
- `src/cleaning/cleaner.py`: cleaning and normalization logic derived from the cleaning notebook.
- `src/validation/validator.py`: quality and consistency checks.
- `src/analytics/transformer.py`: transformation scenarios and chart generation derived from the analytics notebook.
- `src/pipeline/pipeline_runner.py`: end-to-end orchestrator.

## Input and Output
- Input: `data/messy_dataset.csv`
- Cleaned output: `data/cleaned_dataset.csv`
- Validation report: `docs/validation_report.csv`
- Charts: `visualizations/charts`
