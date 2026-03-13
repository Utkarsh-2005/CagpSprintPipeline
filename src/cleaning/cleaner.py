import re
from typing import Dict

import numpy as np
import pandas as pd


USD_TO_INR = 92
EUR_TO_INR = 106


def load_messy_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_cleaned_dataset(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def fix_invalid_minutes(ts):
    if pd.isna(ts):
        return ts

    text = str(ts)
    match = re.search(r"(\d{1,2}):(\d{2})", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))

        if minute > 59:
            hour += minute // 60
            minute = minute % 60

        if hour > 23:
            hour = hour % 24

        text = re.sub(r"\d{1,2}:\d{2}", f"{hour}:{minute:02d}", text, count=1)

    return text


def clean_odometer(value):
    if pd.isna(value):
        return pd.NA

    value = str(value).replace(",", "")
    value = re.sub(r"[^0-9.]", "", value)

    if value == "":
        return pd.NA

    return float(value)


def clean_rate(value):
    if pd.isna(value):
        return pd.NA

    value = str(value).replace(",", "").strip()

    if "USD" in value or "$" in value:
        nums = re.findall(r"\d+", value)
        if nums:
            return int(nums[0]) * USD_TO_INR

    if "EUR" in value or "€" in value:
        nums = re.findall(r"\d+", value)
        if nums:
            return int(nums[0]) * EUR_TO_INR

    nums = re.findall(r"\d+", value)
    if nums:
        return int(nums[0])

    return pd.NA


def validate_license(value):
    if pd.isna(value):
        return None

    value = str(value)
    if re.match(r"DL-\d{10}$", value):
        return value
    return None


def mask_license(value):
    if pd.isna(value) or value is None:
        return None

    return str(value)[:7] + "XXXXXX"


def classify_speed(speed):
    if pd.isna(speed):
        return "Unknown"
    if speed <= 80:
        return "Normal Driving"
    if speed <= 100:
        return "Fast Driving"
    return "Speeding"


def _fill_datetime_with_mean(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        return

    df[column] = pd.to_datetime(df[column], errors="coerce")
    if df[column].notna().any():
        mean_value = df[column].mean()
        df[column] = df[column].fillna(mean_value)


def _safe_mode(series: pd.Series):
    mode = series.mode(dropna=True)
    if len(mode) == 0:
        return pd.NA
    return mode.iloc[0]


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Vehicle_ID" in df.columns:
        df["Vehicle_ID"] = df["Vehicle_ID"].astype("string").str.strip()
        df["Vehicle_ID"] = df["Vehicle_ID"].str.replace("--", "-", regex=False)
        df["Vehicle_ID"] = df["Vehicle_ID"].str.replace("_", "-", regex=False)
        df["Vehicle_ID"] = df["Vehicle_ID"].str.replace(r"([A-Za-z])\s+(\d)", r"\1-\2", regex=True)
        df["Vehicle_ID"] = df["Vehicle_ID"].str.replace(r"([A-Za-z])(\d)", r"\1-\2", regex=True)
        df["Vehicle_ID"] = df["Vehicle_ID"].str.upper()
        df["Vehicle_ID"] = df["Vehicle_ID"].replace("CAR-004", "CAR-04")

        unknown_mask = df["Vehicle_ID"].eq("UNKNOWN")
        df.loc[unknown_mask, "Vehicle_ID"] = pd.NA
        df["Vehicle_ID_Invalid"] = df["Vehicle_ID"].isna()

    for col in ["Pickup_TS", "Return_TS", "Booking_TS"]:
        if col in df.columns:
            df[col] = df[col].apply(fix_invalid_minutes)
            df[col] = (
                df[col]
                .astype("string")
                .str.replace("/", "-", regex=False)
                .str.replace("T", " ", regex=False)
            )
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    for col in ["Odo_Start", "Odo_End"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_odometer)

    if "Fuel_Level" in df.columns:
        df["Fuel_Level"] = df["Fuel_Level"].astype(str).str.replace("%", "", regex=False)
        df["Fuel_Level"] = pd.to_numeric(df["Fuel_Level"], errors="coerce")
        df.loc[df["Fuel_Level"] > 1, "Fuel_Level"] = df["Fuel_Level"] / 100

    if "Rate" in df.columns:
        df["Rate"] = df["Rate"].apply(clean_rate)

    if "City" in df.columns:
        df["City"] = df["City"].astype("string").str.strip().str.lower()
        df["City"] = df["City"].replace("unknown", pd.NA)

        city_mapping: Dict[str, str] = {
            "blr": "bengaluru",
            "bangalore": "bengaluru",
            "bengaluru": "bengaluru",
            "mum": "mumbai",
            "bombay": "mumbai",
            "mumbai": "mumbai",
            "del": "delhi",
            "delhi": "delhi",
            "new delhi": "delhi",
            "chn": "chennai",
            "chennai": "chennai",
        }
        df["City"] = df["City"].replace(city_mapping).str.title()

    if "Reservation_ID" in df.columns:
        df = df.drop_duplicates(subset="Reservation_ID", keep="first")

    if "Return_TS" in df.columns and "Pickup_TS" in df.columns:
        swap_mask = df["Return_TS"] < df["Pickup_TS"]
        df.loc[swap_mask, ["Pickup_TS", "Return_TS"]] = df.loc[swap_mask, ["Return_TS", "Pickup_TS"]].values

        equal_mask = df["Return_TS"] == df["Pickup_TS"]
        df.loc[equal_mask, ["Pickup_TS", "Return_TS"]] = pd.NaT

        df["Duration_Hours"] = (df["Return_TS"] - df["Pickup_TS"]).dt.total_seconds() / 3600

    if "Payment" in df.columns:
        df["Payment"] = df["Payment"].astype(str).str.lower().str.strip()
        payment_dict = {
            "upi": "UPI",
            "credit card": "CARD",
            "debit card": "CARD",
            "card": "CARD",
            "netbanking": "CARD",
            "cash": "CASH",
            "wallet": "WALLET",
            "-": pd.NA,
            "nan": pd.NA,
        }
        df["Payment"] = df["Payment"].replace(payment_dict)

    if "Odo_Start" in df.columns and "Odo_End" in df.columns:
        odo_swap_mask = df["Odo_End"] < df["Odo_Start"]
        df.loc[odo_swap_mask, ["Odo_Start", "Odo_End"]] = df.loc[odo_swap_mask, ["Odo_End", "Odo_Start"]].values

        valid_odo_mask = df["Odo_End"] >= df["Odo_Start"]
        df["Distance_Driven"] = pd.NA
        df.loc[valid_odo_mask, "Distance_Driven"] = (
            df.loc[valid_odo_mask, "Odo_End"] - df.loc[valid_odo_mask, "Odo_Start"]
        )

        if "Vehicle_ID" in df.columns and "Fuel_Level" in df.columns:
            sorted_idx = df.sort_values(["Vehicle_ID", "Odo_Start"], na_position="last").index
            fuel_diff = df.loc[sorted_idx].groupby("Vehicle_ID", dropna=False)["Fuel_Level"].diff()
            refuel_flag = (fuel_diff > 0).map({True: "Refueled", False: "No Refuel"})

            df["Refuel_Event"] = pd.NA
            df.loc[refuel_flag.index, "Refuel_Event"] = refuel_flag
            df.loc[~valid_odo_mask, "Refuel_Event"] = pd.NA

    if "Vehicle_ID" in df.columns and "Pickup_TS" in df.columns and "Return_TS" in df.columns:
        df = df.sort_values(["Vehicle_ID", "Pickup_TS"], na_position="last")
        df["Prev_Return"] = df.groupby("Vehicle_ID", dropna=False)["Return_TS"].shift(1)
        df["Overlap_Flag"] = df["Pickup_TS"] < df["Prev_Return"]

    if "Driver_License" in df.columns:
        df["Driver_License"] = df["Driver_License"].apply(validate_license)
        df["Driver_License"] = df["Driver_License"].apply(mask_license)

    promo_expiry = {
        "NEW10": pd.to_datetime("2026-03-31"),
        "DISC20": pd.to_datetime("2026-02-28"),
        "SAVE50": pd.to_datetime("2026-06-30"),
        "WELCOME5": pd.to_datetime("2026-01-31"),
    }
    if "Promo_Code" in df.columns:
        df["Promo_Expiry"] = df["Promo_Code"].map(promo_expiry)
        df["Promo_Status"] = "Valid"
        df.loc[df["Promo_Code"].notna() & df["Promo_Expiry"].isna(), "Promo_Status"] = "Invalid_Code"

        if "Pickup_TS" in df.columns:
            df.loc[
                df["Promo_Expiry"].notna() & (df["Pickup_TS"] > df["Promo_Expiry"]),
                "Promo_Status",
            ] = "Expired"

    for geo_col in ["GPS_Lat", "GPS_Lon"]:
        if geo_col in df.columns:
            df[geo_col] = pd.to_numeric(
                df[geo_col].astype(str).str.strip().str.replace(r"^'+|'+$", "", regex=True),
                errors="coerce",
            )
            df[geo_col] = df[geo_col].interpolate()
            df[geo_col] = df[geo_col].rolling(window=3, min_periods=1).mean().round(4)

    if "Speed" in df.columns:
        df["Speed"] = df["Speed"].astype(str)
        df["Speed"] = df["Speed"].str.replace("km/h", "", regex=False)
        df["Speed"] = df["Speed"].str.replace("kmh", "", regex=False)
        df["Speed"] = df["Speed"].replace("fast", 90)
        df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce")
        df["Driver_Behavior"] = df["Speed"].apply(classify_speed)

    if "Notes" in df.columns:
        df["Notes"] = df["Notes"].astype("string")
        df["Notes"] = df["Notes"].str.replace(r"\b\d{10}\b", "[REDACTED]", regex=True)
        df["Notes"] = df["Notes"].str.replace(r"\S+@\S+", "[REDACTED]", regex=True)
        df["Notes"] = df["Notes"].str.replace(r"\b[A-Z0-9]{8,}\b", "[REDACTED]", regex=True)

    if "Rate" in df.columns:
        df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")

    if "Rate" in df.columns:
        df["Total_Amount"] = (df["Rate"] * 1.18).round(2)

    _fill_datetime_with_mean(df, "Pickup_TS")
    _fill_datetime_with_mean(df, "Return_TS")
    _fill_datetime_with_mean(df, "Booking_TS")

    if "Fuel_Level" in df.columns:
        df["Fuel_Level"] = df["Fuel_Level"].fillna(df["Fuel_Level"].mean())
    if "Rate" in df.columns:
        df["Rate"] = df["Rate"].fillna(df["Rate"].mean())
    if "Promo_Code" in df.columns:
        df["Promo_Code"] = df["Promo_Code"].fillna(_safe_mode(df["Promo_Code"]))
    if "City" in df.columns:
        df["City"] = df["City"].fillna(_safe_mode(df["City"]))
    if "Speed" in df.columns:
        df["Speed"] = df["Speed"].fillna(df["Speed"].median())
    if "Payment" in df.columns:
        df["Payment"] = df["Payment"].fillna(_safe_mode(df["Payment"]))
    if "Damage_Flag" in df.columns:
        df["Damage_Flag"] = df["Damage_Flag"].fillna("None")
    if "Notes" in df.columns:
        df["Notes"] = df["Notes"].fillna("No Notes")
    if "Duration_Hours" in df.columns:
        df["Duration_Hours"] = pd.to_numeric(df["Duration_Hours"], errors="coerce").round(2)

    if "Booking_TS" in df.columns and "Pickup_TS" in df.columns:
        mask_booking_after_pickup = df["Booking_TS"] > df["Pickup_TS"]
        df.loc[mask_booking_after_pickup, "Booking_TS"] = df["Pickup_TS"] - pd.Timedelta(days=1)

    if "Booking_Status" in df.columns and "Odo_End" in df.columns and "Odo_Start" in df.columns:
        cancelled_mask = df["Booking_Status"] == "Cancelled"
        df.loc[cancelled_mask, "Odo_End"] = df["Odo_Start"]

    if "Booking_Status" in df.columns and "Total_Amount" in df.columns:
        cancelled_mask = df["Booking_Status"] == "Cancelled"
        df.loc[cancelled_mask, "Total_Amount"] = 0
        df["Total_Amount"] = df["Total_Amount"].fillna(df["Total_Amount"].mean())

    for col in ["Fuel_Level", "Rate", "Total_Amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    if "Distance_Driven" in df.columns and "Odo_Start" in df.columns and "Odo_End" in df.columns:
        df["Distance_Driven"] = df["Odo_End"] - df["Odo_Start"]

    if "Vehicle_ID" in df.columns:
        df["Vehicle_ID"] = df["Vehicle_ID"].fillna("CAR-600")

    if "Promo_Expiry" in df.columns and "Promo_Status" in df.columns:
        df.loc[df["Promo_Expiry"].isna(), "Promo_Status"] = "Invalid"

    return df
