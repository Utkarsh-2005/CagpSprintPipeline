from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.analytics.transformer import run_transformations
from src.cleaning.cleaner import clean_dataset, load_messy_dataset, save_cleaned_dataset
from src.validation.validator import run_validation_checks


class CarManufacturingPipeline:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.docs_dir = self.project_root / "docs"
        self.charts_dir = self.project_root / "visualizations" / "charts"

        self.messy_data_path = self.data_dir / "messy_dataset.csv"
        self.cleaned_data_path = self.data_dir / "cleaned_dataset.csv"
        self.validation_report_path = self.docs_dir / "validation_report.csv"

    def run(self) -> Dict[str, Any]:
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        raw_df = load_messy_dataset(str(self.messy_data_path))
        cleaned_df = clean_dataset(raw_df)
        save_cleaned_dataset(cleaned_df, str(self.cleaned_data_path))

        validation_report = run_validation_checks(cleaned_df)
        pd.DataFrame([validation_report]).to_csv(self.validation_report_path, index=False)

        analytics_results = run_transformations(self.cleaned_data_path, self.charts_dir)

        return {
            "input_rows": int(len(raw_df)),
            "cleaned_rows": int(len(cleaned_df)),
            "cleaned_dataset_path": str(self.cleaned_data_path),
            "validation_report_path": str(self.validation_report_path),
            "charts_dir": str(self.charts_dir),
            "analytics_result_count": len(analytics_results),
            "validation_report": validation_report,
        }
