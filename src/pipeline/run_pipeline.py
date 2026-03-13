from pathlib import Path

from src.pipeline.pipeline_runner import CarManufacturingPipeline


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    pipeline = CarManufacturingPipeline(project_root)
    summary = pipeline.run()

    print("Pipeline completed successfully")
    print(f"Input rows: {summary['input_rows']}")
    print(f"Cleaned rows: {summary['cleaned_rows']}")
    print(f"Cleaned dataset: {summary['cleaned_dataset_path']}")
    print(f"Validation report: {summary['validation_report_path']}")
    print(f"Charts folder: {summary['charts_dir']}")
    print(f"Analytics scenarios generated: {summary['analytics_result_count']}")


if __name__ == "__main__":
    main()
