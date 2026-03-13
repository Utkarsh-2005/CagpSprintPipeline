from typing import Dict, List

import pandas as pd


REQUIRED_COLUMNS = [
    "Reservation_ID",
    "Customer_ID",
    "Vehicle_ID",
    "Booking_Status",
    "Booking_TS",
    "Pickup_TS",
    "Return_TS",
    "Odo_Start",
    "Odo_End",
    "Fuel_Level",
    "Rate",
    "City",
    "Speed",
    "Total_Amount",
]


def _count_if(df: pd.DataFrame, expr: pd.Series) -> int:
    if expr is None:
        return 0
    return int(expr.fillna(False).sum())


def run_validation_checks(df: pd.DataFrame) -> Dict[str, int | List[str]]:
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    vehicle_id_pattern_violations = 0
    if "Vehicle_ID" in df.columns:
        non_null_vehicle_ids = df["Vehicle_ID"].dropna().astype(str)
        vehicle_id_pattern_violations = int((~non_null_vehicle_ids.str.match(r"^CAR-\d{2,3}$")).sum())

    report: Dict[str, int | List[str]] = {
        "row_count": int(len(df)),
        "missing_required_columns": missing_columns,
        "vehicle_id_pattern_violations": vehicle_id_pattern_violations,
        "negative_duration_rows": _count_if(df, df.get("Duration_Hours", pd.Series(dtype="float")).lt(0)),
        "return_before_pickup_rows": _count_if(
            df,
            (pd.to_datetime(df.get("Return_TS"), errors="coerce") < pd.to_datetime(df.get("Pickup_TS"), errors="coerce")),
        ) if "Return_TS" in df.columns and "Pickup_TS" in df.columns else 0,
        "odometer_inversion_rows": _count_if(df, df.get("Odo_End", pd.Series(dtype="float")).lt(df.get("Odo_Start", pd.Series(dtype="float")))),
        "fuel_out_of_range_rows": _count_if(
            df,
            (pd.to_numeric(df.get("Fuel_Level"), errors="coerce") < 0)
            | (pd.to_numeric(df.get("Fuel_Level"), errors="coerce") > 1),
        ) if "Fuel_Level" in df.columns else 0,
        "null_vehicle_id_rows": int(df["Vehicle_ID"].isna().sum()) if "Vehicle_ID" in df.columns else 0,
        "null_city_rows": int(df["City"].isna().sum()) if "City" in df.columns else 0,
        "invalid_promo_rows": _count_if(
            df,
            df.get("Promo_Status", pd.Series(dtype="object")).isin(["Invalid", "Invalid_Code"]),
        ) if "Promo_Status" in df.columns else 0,
    }

    return report
