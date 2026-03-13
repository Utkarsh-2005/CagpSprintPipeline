from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def _prepare_dataframe(cleaned_dataset_path: str | Path) -> tuple[pd.DataFrame, float]:
    df = pd.read_csv(cleaned_dataset_path)

    required_defaults = {
        "Reservation_ID": "",
        "Customer_ID": "",
        "Vehicle_ID": "",
        "Vehicle_Class": "Unknown",
        "Booking_Status": "Unknown",
        "Booking_TS": pd.NaT,
        "Pickup_TS": pd.NaT,
        "Return_TS": pd.NaT,
        "Odo_Start": np.nan,
        "Odo_End": np.nan,
        "Fuel_Level": np.nan,
        "Rate": np.nan,
        "Promo_Code": "",
        "City": "Unknown",
        "GPS_Lat": np.nan,
        "GPS_Lon": np.nan,
        "Speed": np.nan,
        "Damage_Flag": "None",
        "Notes": "",
        "Vehicle_ID_Invalid": False,
        "Duration_Hours": np.nan,
        "Distance_Driven": np.nan,
        "Refuel_Event": "",
        "Driver_Behavior": "Unknown",
        "Total_Amount": np.nan,
    }
    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    for col in ["Booking_TS", "Pickup_TS", "Return_TS", "Prev_Return", "Promo_Expiry"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in [
        "Duration_Hours",
        "Distance_Driven",
        "Odo_Start",
        "Odo_End",
        "Rate",
        "Total_Amount",
        "Fuel_Level",
        "Speed",
        "GPS_Lat",
        "GPS_Lon",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    bool_values = df["Vehicle_ID_Invalid"].astype(str).str.lower().str.strip()
    df["vehicle_id_invalid_flag"] = bool_values.isin(["true", "1", "yes"])

    df["is_completed"] = df["Booking_Status"].eq("Completed")
    df["rental_hours"] = df["Duration_Hours"]

    mask_missing_hours = df["rental_hours"].isna() & df["Pickup_TS"].notna() & df["Return_TS"].notna()
    df.loc[mask_missing_hours, "rental_hours"] = (
        (df.loc[mask_missing_hours, "Return_TS"] - df.loc[mask_missing_hours, "Pickup_TS"]).dt.total_seconds() / 3600
    )
    df["rental_hours"] = df["rental_hours"].clip(lower=0)

    df["distance_km"] = df["Distance_Driven"].clip(lower=0)
    df["lead_time_hours"] = ((df["Pickup_TS"] - df["Booking_TS"]).dt.total_seconds() / 3600).clip(lower=0)
    df["booking_month"] = df["Booking_TS"].dt.to_period("M").astype(str)
    df["pickup_date"] = df["Pickup_TS"].dt.date

    a_start = df["Pickup_TS"].min()
    a_end = df["Return_TS"].max()
    analysis_hours = max((a_end - a_start).total_seconds() / 3600, 1) if pd.notna(a_start) and pd.notna(a_end) else 1

    return df, analysis_hours


def _save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_transformations(cleaned_dataset_path: str | Path, charts_dir: str | Path) -> Dict[str, pd.DataFrame]:
    charts_path = Path(charts_dir)
    charts_path.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    pd.set_option("display.max_columns", 120)

    df, analysis_hours = _prepare_dataframe(cleaned_dataset_path)
    results: Dict[str, pd.DataFrame] = {}

    fleet = df.groupby(["City", "Vehicle_Class"], dropna=False)["Vehicle_ID"].nunique().rename("available_cars")
    rental = (
        df.loc[df["is_completed"]]
        .groupby(["City", "Vehicle_Class"], dropna=False)["rental_hours"]
        .sum()
        .rename("rental_hours")
    )
    util = pd.concat([fleet, rental], axis=1).fillna(0).reset_index()
    util["fleet_hours"] = util["available_cars"] * analysis_hours
    util["utilization"] = np.where(util["fleet_hours"] > 0, util["rental_hours"] / util["fleet_hours"], 0)
    util = util.sort_values("utilization", ascending=False)
    results["scenario_01_utilization"] = util

    if not util.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=util.head(15), x="utilization", y="Vehicle_Class", hue="City")
        plt.title("Top Utilization by City and Vehicle Class")
        _save_plot(charts_path / "01_utilization_top.png")

    completed = df[df["is_completed"]].copy()
    rev = completed.groupby(["City", "Vehicle_Class"], dropna=False)["Total_Amount"].sum().rename("revenue")
    hours = completed.groupby(["City", "Vehicle_Class"], dropna=False)["rental_hours"].sum().rename("rental_hours")
    base = (
        completed.assign(base_bill=completed["Rate"] * completed["rental_hours"])
        .groupby(["City", "Vehicle_Class"], dropna=False)["base_bill"]
        .sum()
    )

    rev_metrics = util.set_index(["City", "Vehicle_Class"])[["available_cars"]].join(rev).join(hours).join(base).fillna(0)
    rev_metrics = rev_metrics.rename(columns={"base_bill": "base_revenue_proxy"}).reset_index()
    rev_metrics["RevPAC"] = np.where(rev_metrics["available_cars"] > 0, rev_metrics["revenue"] / rev_metrics["available_cars"], 0)
    rev_metrics["yield_per_hour"] = np.where(rev_metrics["rental_hours"] > 0, rev_metrics["revenue"] / rev_metrics["rental_hours"], 0)
    rev_metrics["realization_vs_base"] = np.where(
        rev_metrics["base_revenue_proxy"] > 0,
        rev_metrics["revenue"] / rev_metrics["base_revenue_proxy"],
        np.nan,
    )
    results["scenario_02_revpac"] = rev_metrics

    trip = df[(df["is_completed"]) & (df["distance_km"] > 0)].copy()
    trip["cost_per_km"] = trip["Total_Amount"] / trip["distance_km"]
    distance_cost = (
        trip.groupby(["City", "Vehicle_Class"], dropna=False)
        .agg(
            total_distance_km=("distance_km", "sum"),
            avg_cost_per_km=("cost_per_km", "mean"),
            median_cost_per_km=("cost_per_km", "median"),
        )
        .reset_index()
        .sort_values("total_distance_km", ascending=False)
    )
    results["scenario_03_distance_cost"] = distance_cost

    if len(trip) > 0:
        plt.figure(figsize=(10, 5))
        sns.scatterplot(
            data=trip.sample(min(len(trip), 2000), random_state=42),
            x="distance_km",
            y="cost_per_km",
            hue="Vehicle_Class",
            alpha=0.5,
        )
        plt.title("Cost per KM vs Distance")
        _save_plot(charts_path / "03_cost_per_km_scatter.png")

    seq = df.sort_values(["Vehicle_ID", "Pickup_TS"]).copy()
    seq["prev_return_ts"] = seq.groupby("Vehicle_ID")["Return_TS"].shift(1)
    seq["prev_city"] = seq.groupby("Vehicle_ID")["City"].shift(1)
    seq["idle_hours"] = ((seq["Pickup_TS"] - seq["prev_return_ts"]).dt.total_seconds() / 3600).clip(lower=0)
    seq["repositioned"] = (seq["City"] != seq["prev_city"]) & seq["prev_city"].notna()

    idle_summary = (
        seq.groupby("Vehicle_Class", dropna=False)
        .agg(
            avg_idle_hours=("idle_hours", "mean"),
            median_idle_hours=("idle_hours", "median"),
            reposition_events=("repositioned", "sum"),
        )
        .reset_index()
        .sort_values("avg_idle_hours", ascending=False)
    )
    results["scenario_04_idle_reposition"] = idle_summary

    pricing = df.copy()
    pricing["month"] = pricing["Pickup_TS"].dt.month
    pricing["weekday"] = pricing["Pickup_TS"].dt.day_name()
    pricing["daily_demand"] = pricing.groupby(["City", "pickup_date"], dropna=False)["Reservation_ID"].transform("count")
    results["scenario_05_pricing_features"] = pricing[
        ["Reservation_ID", "City", "Vehicle_Class", "Rate", "lead_time_hours", "daily_demand", "month", "weekday"]
    ].copy()

    if len(pricing) > 0:
        plt.figure(figsize=(10, 5))
        sns.scatterplot(
            data=pricing.sample(min(len(pricing), 2000), random_state=42),
            x="lead_time_hours",
            y="Rate",
            hue="City",
            alpha=0.4,
        )
        plt.title("Lead Time vs Rate")
        _save_plot(charts_path / "05_lead_time_vs_rate.png")

    fuel = df[(df["is_completed"]) & (df["distance_km"] > 0)].copy()
    fuel["fuel_used_proxy"] = (1 - fuel["Fuel_Level"].fillna(0.5)).clip(lower=0.05, upper=1.0)
    fuel["km_per_fuel_proxy"] = fuel["distance_km"] / fuel["fuel_used_proxy"]
    fuel_eff = (
        fuel.groupby("Vehicle_Class", dropna=False)
        .agg(
            trips=("Reservation_ID", "count"),
            avg_km_per_fuel_proxy=("km_per_fuel_proxy", "mean"),
            median_km_per_fuel_proxy=("km_per_fuel_proxy", "median"),
        )
        .reset_index()
        .sort_values("avg_km_per_fuel_proxy", ascending=False)
    )
    results["scenario_06_fuel_efficiency"] = fuel_eff

    damage = df.copy()
    damage["damage_event"] = damage["Damage_Flag"].fillna("None").ne("None")
    damage_rate = (
        damage.groupby(["City", "Vehicle_Class"], dropna=False)
        .agg(
            rentals=("Reservation_ID", "count"),
            damage_events=("damage_event", "sum"),
        )
        .reset_index()
    )
    damage_rate["damage_per_100_rentals"] = np.where(
        damage_rate["rentals"] > 0,
        100 * damage_rate["damage_events"] / damage_rate["rentals"],
        0,
    )
    results["scenario_07_damage_rate"] = damage_rate

    cust = df.copy()
    cust["booking_month_period"] = cust["Booking_TS"].dt.to_period("M")
    cohort_month = cust.groupby("Customer_ID")["booking_month_period"].min().rename("cohort_month")
    cust = cust.join(cohort_month, on="Customer_ID")

    valid = cust.dropna(subset=["booking_month_period", "cohort_month"]).copy()
    valid["cohort_index"] = valid["booking_month_period"].astype(int) - valid["cohort_month"].astype(int)

    ret_counts = valid.groupby(["cohort_month", "cohort_index"])["Customer_ID"].nunique().unstack(fill_value=0)
    if not ret_counts.empty:
        base = ret_counts.iloc[:, 0]
        retention = ret_counts.div(base, axis=0).round(3)
    else:
        retention = pd.DataFrame()

    valid["nps_bucket"] = np.select(
        [
            (valid["Driver_Behavior"].eq("Normal Driving") & valid["Damage_Flag"].eq("None")),
            (valid["Driver_Behavior"].isin(["Speeding", "Fast Driving"]) | valid["Damage_Flag"].eq("Major")),
        ],
        ["Promoter", "Detractor"],
        default="Passive",
    )

    nps_rollup = valid.groupby("cohort_month")["nps_bucket"].value_counts(normalize=True).unstack(fill_value=0)
    nps_rollup["NPS"] = (nps_rollup.get("Promoter", 0) - nps_rollup.get("Detractor", 0)) * 100

    results["scenario_08_retention"] = retention.reset_index() if not retention.empty else retention
    results["scenario_08_nps"] = nps_rollup.reset_index()

    fraud = df.copy()
    fraud["short_return_flag"] = fraud["is_completed"] & (fraud["rental_hours"] < 2)
    fraud["odo_diff"] = fraud["Odo_End"] - fraud["Odo_Start"]
    fraud["odo_anomaly_flag"] = (fraud["odo_diff"] < 0) | ((fraud["odo_diff"] - fraud["distance_km"]).abs() > 50)
    fraud["speed_risk_flag"] = fraud["Speed"] > 120
    fraud["fraud_risk_score"] = (
        30 * fraud["short_return_flag"].astype(int)
        + 50 * fraud["odo_anomaly_flag"].fillna(False).astype(int)
        + 20 * fraud["speed_risk_flag"].fillna(False).astype(int)
        + 20 * fraud["vehicle_id_invalid_flag"].astype(int)
    ).clip(0, 100)
    fraud_view = fraud[
        [
            "Reservation_ID",
            "Customer_ID",
            "Vehicle_ID",
            "City",
            "fraud_risk_score",
            "short_return_flag",
            "odo_anomaly_flag",
            "speed_risk_flag",
            "vehicle_id_invalid_flag",
        ]
    ]
    results["scenario_09_fraud_risk"] = fraud_view.sort_values("fraud_risk_score", ascending=False)

    latest = df.sort_values("Return_TS").groupby("Vehicle_ID", as_index=False).tail(1).copy()
    latest["current_odometer"] = latest["Odo_End"].fillna(latest["Odo_Start"])
    latest["km_to_next_service"] = 10000 - (latest["current_odometer"] % 10000)
    latest["days_since_last_return"] = (pd.Timestamp.today().normalize() - latest["Return_TS"]).dt.days
    latest["maintenance_due"] = (latest["km_to_next_service"] <= 500) | (latest["days_since_last_return"] >= 180)
    latest["maintenance_priority"] = (
        ((500 - latest["km_to_next_service"]).clip(lower=0) / 500) * 0.6
        + (latest["days_since_last_return"].fillna(0).clip(lower=0) / 180).clip(upper=2) * 0.4
    )
    results["scenario_10_maintenance"] = latest[
        [
            "Vehicle_ID",
            "Vehicle_Class",
            "current_odometer",
            "km_to_next_service",
            "days_since_last_return",
            "maintenance_due",
            "maintenance_priority",
        ]
    ].sort_values(["maintenance_due", "maintenance_priority"], ascending=[False, False])

    overstay = df[df["is_completed"]].copy()
    overstay["expected_hours_from_bill"] = np.where(overstay["Rate"] > 0, overstay["Total_Amount"] / overstay["Rate"], np.nan)
    overstay["overstay_hours"] = (overstay["rental_hours"] - overstay["expected_hours_from_bill"]).clip(lower=0)
    overstay["overstay_penalty"] = overstay["overstay_hours"] * 0.25 * overstay["Rate"]
    results["scenario_11_overstay"] = overstay[
        [
            "Reservation_ID",
            "City",
            "Vehicle_Class",
            "rental_hours",
            "expected_hours_from_bill",
            "overstay_hours",
            "overstay_penalty",
        ]
    ].sort_values("overstay_penalty", ascending=False)

    punct = df.copy()
    punct["scheduled_pickup"] = punct["Booking_TS"] + pd.Timedelta(hours=24)
    punct["pickup_delay_min"] = (punct["Pickup_TS"] - punct["scheduled_pickup"]).dt.total_seconds() / 60
    punct["actual_duration_hours"] = (punct["Return_TS"] - punct["Pickup_TS"]).dt.total_seconds() / 3600
    punct["return_delay_min"] = (punct["actual_duration_hours"] - punct["Duration_Hours"]).fillna(0) * 60
    punct["pickup_on_time"] = punct["pickup_delay_min"].abs() <= 30
    punct["return_on_time"] = punct["return_delay_min"].abs() <= 30

    punct_stats = (
        punct.groupby("City", dropna=False)
        .agg(
            pickup_on_time_rate=("pickup_on_time", "mean"),
            return_on_time_rate=("return_on_time", "mean"),
            avg_pickup_delay_min=("pickup_delay_min", "mean"),
            avg_return_delay_min=("return_delay_min", "mean"),
        )
        .reset_index()
    )
    punct_stats["pickup_on_time_rate"] = (punct_stats["pickup_on_time_rate"] * 100).round(2)
    punct_stats["return_on_time_rate"] = (punct_stats["return_on_time_rate"] * 100).round(2)
    results["scenario_12_punctuality"] = punct_stats

    geo = df.dropna(subset=["GPS_Lat", "GPS_Lon"]).copy()
    if len(geo) > 0:
        plt.figure(figsize=(8, 6))
        plt.hexbin(geo["GPS_Lon"], geo["GPS_Lat"], gridsize=30, cmap="YlOrRd", mincnt=1)
        plt.colorbar(label="Booking density")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Pickup Hotspot Hexbin")
        _save_plot(charts_path / "13_pickup_hotspot_hexbin.png")

    hotspots = (
        geo.assign(lat_bin=geo["GPS_Lat"].round(2), lon_bin=geo["GPS_Lon"].round(2))
        .groupby(["City", "lat_bin", "lon_bin"], dropna=False)
        .size()
        .reset_index(name="bookings")
        .sort_values("bookings", ascending=False)
    )
    results["scenario_13_geo_hotspots"] = hotspots

    upsell = df.copy()
    notes = upsell["Notes"].fillna("").str.lower()
    upsell["addon_navigation"] = notes.str.contains("navigation")
    upsell["addon_cleaning"] = notes.str.contains("clean")
    upsell["addon_fuel_plan"] = notes.str.contains("fuel") | (upsell["Fuel_Level"].fillna(0.5) < 0.25)
    upsell["addon_protection"] = upsell["Driver_Behavior"].isin(["Speeding", "Fast Driving"]) | upsell["Damage_Flag"].isin(["Minor", "Major"])
    flag_cols = ["addon_navigation", "addon_cleaning", "addon_fuel_plan", "addon_protection"]
    upsell["upsell_flag_count"] = upsell[flag_cols].sum(axis=1)
    upsell["upsell_opportunity"] = upsell["upsell_flag_count"] > 0

    upsell_summary = (
        upsell.groupby("City", dropna=False)
        .agg(
            rentals=("Reservation_ID", "count"),
            opportunities=("upsell_opportunity", "sum"),
        )
        .reset_index()
    )
    upsell_summary["opportunity_rate"] = 100 * upsell_summary["opportunities"] / upsell_summary["rentals"]
    results["scenario_14_upsell"] = upsell_summary.sort_values("opportunity_rate", ascending=False)

    cancel = df.copy()
    cancel["is_cancelled"] = cancel["Booking_Status"].isin(["Cancelled", "No_Show"])
    notes = cancel["Notes"].fillna("").str.lower()
    cancel["cancel_reason_bucket"] = np.select(
        [
            notes.str.contains("traffic"),
            notes.str.contains("fuel"),
            notes.str.contains("clean"),
            notes.str.contains("scratch|damage"),
            notes.str.contains("early pickup"),
            notes.str.contains("no notes"),
        ],
        ["Traffic", "Fuel", "Cleaning", "Damage Concern", "Schedule Change", "Unspecified"],
        default="Other",
    )

    cancel_rate = (
        cancel.groupby(["City", "Vehicle_Class"], dropna=False)
        .agg(total_bookings=("Reservation_ID", "count"), cancelled=("is_cancelled", "sum"))
        .reset_index()
    )
    cancel_rate["cancellation_rate"] = 100 * cancel_rate["cancelled"] / cancel_rate["total_bookings"]
    reason_dist = (
        cancel[cancel["is_cancelled"]]
        .groupby("cancel_reason_bucket")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    results["scenario_15_cancel_rate"] = cancel_rate
    results["scenario_15_cancel_reasons"] = reason_dist

    driver = df.copy()
    driver["driver_score"] = 100
    driver["driver_score"] = driver["driver_score"] - np.where(driver["Speed"] > 120, 30, np.where(driver["Speed"] > 100, 15, 0))
    driver["driver_score"] = driver["driver_score"] - np.where(
        driver["Driver_Behavior"].eq("Speeding"),
        25,
        np.where(driver["Driver_Behavior"].eq("Fast Driving"), 10, 0),
    )
    driver["driver_score"] = driver["driver_score"] - np.where(
        driver["Damage_Flag"].eq("Major"),
        20,
        np.where(driver["Damage_Flag"].eq("Minor"), 8, 0),
    )
    driver["driver_score"] = driver["driver_score"].clip(lower=0, upper=100)

    driver_summary = (
        driver.groupby("Customer_ID", dropna=False)
        .agg(
            trips=("Reservation_ID", "count"),
            avg_driver_score=("driver_score", "mean"),
        )
        .reset_index()
        .sort_values("avg_driver_score")
    )
    results["scenario_16_driver_scoring"] = driver_summary

    mix = util.merge(rev_metrics[["City", "Vehicle_Class", "RevPAC"]], on=["City", "Vehicle_Class"], how="left")
    mix["fleet_share"] = mix.groupby("City")["available_cars"].transform(lambda s: s / max(s.sum(), 1))
    mix["util_norm"] = mix.groupby("City")["utilization"].transform(lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9))
    mix["rev_norm"] = mix.groupby("City")["RevPAC"].transform(lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9))
    mix["mix_score"] = 0.6 * mix["util_norm"] + 0.4 * mix["rev_norm"]
    mix["mix_action"] = np.where(
        mix["mix_score"] >= 0.66,
        "Increase share",
        np.where(mix["mix_score"] >= 0.33, "Hold / optimize", "Reduce share"),
    )
    results["scenario_17_mix_optimization"] = mix.sort_values(["City", "mix_score"], ascending=[True, False])

    elastic = df.copy()
    elastic["week"] = elastic["Booking_TS"].dt.to_period("W").astype(str)
    weekly = (
        elastic.groupby(["City", "Vehicle_Class", "week"], dropna=False)
        .agg(
            bookings=("Reservation_ID", "count"),
            avg_rate=("Rate", "mean"),
            avg_lead_time=("lead_time_hours", "mean"),
        )
        .reset_index()
    )

    def elasticity_proxy(group: pd.DataFrame):
        group = group[(group["bookings"] > 0) & (group["avg_rate"] > 0)].dropna(subset=["bookings", "avg_rate"])
        if len(group) < 3 or group["avg_rate"].nunique() < 2:
            return np.nan
        x = np.log(group["avg_rate"])
        y = np.log(group["bookings"])
        return np.cov(x, y, bias=True)[0, 1] / (np.var(x) + 1e-9)

    elasticity_table = (
        weekly.groupby(["City", "Vehicle_Class"], dropna=False)
        .apply(elasticity_proxy)
        .reset_index(name="price_elasticity")
    )
    results["scenario_18_elasticity"] = elasticity_table

    health_src = df.copy()
    notes = health_src["Notes"].fillna("").str.lower()
    health_src["fault_points"] = (
        np.where(health_src["Damage_Flag"].eq("Major"), 20, 0)
        + np.where(health_src["Damage_Flag"].eq("Minor"), 10, 0)
        + np.where(notes.str.contains("alert|malfunction|scratch"), 8, 0)
    )
    health_src["usage_hours_component"] = np.where(health_src["is_completed"], health_src["rental_hours"].fillna(0), 0)

    health = (
        health_src.groupby(["Vehicle_ID", "Vehicle_Class"], dropna=False)
        .agg(
            fault_points=("fault_points", "sum"),
            usage_hours=("usage_hours_component", "sum"),
            avg_speed=("Speed", "mean"),
            rentals=("Reservation_ID", "count"),
        )
        .reset_index()
    )
    health["fault_norm"] = health["fault_points"] / (health["fault_points"].max() + 1e-9)
    health["usage_norm"] = health["usage_hours"] / (health["usage_hours"].max() + 1e-9)
    health["speed_norm"] = (health["avg_speed"].fillna(0) / 120).clip(0, 1)
    health["fleet_health_score"] = 100 - (50 * health["fault_norm"] + 30 * health["usage_norm"] + 20 * health["speed_norm"])
    health["fleet_health_score"] = health["fleet_health_score"].clip(0, 100)
    results["scenario_19_fleet_health"] = health.sort_values("fleet_health_score")

    churn_src = df.copy()
    churn_src["booking_date"] = churn_src["Booking_TS"].dt.floor("D")
    cust_counts = churn_src.groupby("Customer_ID")["Reservation_ID"].count().rename("total_bookings")
    subs = cust_counts[cust_counts >= 3].index
    sub_df = churn_src[churn_src["Customer_ID"].isin(subs)].copy()

    snapshot = sub_df["booking_date"].max() + pd.Timedelta(days=1) if not sub_df.empty else pd.Timestamp.today().normalize()
    last_booking = sub_df.groupby("Customer_ID")["booking_date"].max().rename("last_booking")
    sub_df["days_ago"] = (snapshot - sub_df["booking_date"]).dt.days

    recent_90 = (
        sub_df[sub_df["days_ago"] <= 90]
        .groupby("Customer_ID")["Reservation_ID"]
        .count()
        .rename("bookings_recent_90")
    )
    prev_90 = (
        sub_df[(sub_df["days_ago"] > 90) & (sub_df["days_ago"] <= 180)]
        .groupby("Customer_ID")["Reservation_ID"]
        .count()
        .rename("bookings_prev_90")
    )

    churn = pd.concat([cust_counts, last_booking, recent_90, prev_90], axis=1).fillna(0)
    churn = churn.loc[churn.index.isin(subs)].copy()

    churn["days_since_last_booking"] = (snapshot - pd.to_datetime(churn["last_booking"])) .dt.days
    churn["drop_ratio"] = np.where(
        churn["bookings_prev_90"] > 0,
        (churn["bookings_prev_90"] - churn["bookings_recent_90"]) / churn["bookings_prev_90"],
        0,
    )
    churn["recency_norm"] = (churn["days_since_last_booking"] / 90).clip(0, 1)
    churn["drop_norm"] = churn["drop_ratio"].clip(0, 1)
    churn["churn_likelihood"] = (0.6 * churn["recency_norm"] + 0.4 * churn["drop_norm"]).clip(0, 1)
    churn["churn_band"] = pd.cut(
        churn["churn_likelihood"],
        bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
    )
    results["scenario_20_churn"] = churn.sort_values("churn_likelihood", ascending=False)

    return results
