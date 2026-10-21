"""
Forecast logic shared by app.py and daily_job.py (no Streamlit in here).
Relies on red_team.py for the model, features and scenarios.
"""
import numpy as np
import pandas as pd

import red_team as rt


def daily_forecast(model, month_df, temp_shifts):
    """Forecast each day using its own temperature adjustment."""
    frame = month_df.copy()
    frame["temp_whatif"] = month_df["temp_expected"] + temp_shifts
    return model.predict(rt.make_features(frame, "temp_whatif"))


def daily_forecast_table(month_df, temp_shifts, base_daily, whatif_daily):
    """Build a date-by-date view of inputs and adjusted Dth/day forecasts."""
    return pd.DataFrame(
        {
            "Gas Day": month_df["date"].dt.strftime("%Y-%m-%d"),
            "Expected temperature (F)": month_df["temp_expected"].round(1),
            "Temperature shift (F)": temp_shifts,
            "Forecast (Dth/day)": base_daily.round(1),
            "What-if forecast (Dth/day)": whatif_daily.round(1),
            "Change (Dth/day)": (whatif_daily - base_daily).round(1),
        }
    )


def prepare_forecast_month(state_df, month):
    """Include the unobserved remainder of the current month in the forecast."""
    month_df, model = rt.prepare_month(state_df, month)
    period = pd.Period(month)
    if (
        period == pd.Timestamp.now().to_period("M")
        and not month_df.empty
        and month_df["date"].max() < period.end_time
    ):
        future_dates = pd.date_range(
            month_df["date"].max() + pd.Timedelta(days=1),
            period.end_time,
            freq="D",
        )
        future = pd.DataFrame({"date": future_dates})
        future["is_holiday"] = rt.holiday_flags(future["date"])
        future["usage"] = np.nan
        history = state_df[state_df["date"] < period.start_time]
        future["temp_expected"] = rt.expected_temperatures(
            history, future["date"]
        )
        month_df = pd.concat([month_df, future], ignore_index=True)
    return month_df, model


def build_sample_nominations(state_data):
    """Create repeatable illustrative nominations for every state and forecast day."""
    frames = []
    for state_index, (state, (month_df, model)) in enumerate(state_data.items()):
        baseline_daily = daily_forecast(model, month_df, np.zeros(len(month_df)))
        rng = np.random.default_rng(42 + state_index)
        nominated = np.maximum(
            baseline_daily * (1 + rng.uniform(-0.04, 0.04, len(month_df))), 0
        ).round(1)
        frames.append(
            pd.DataFrame(
                {
                    "date": month_df["date"].to_numpy(),
                    "state": state,
                    "nominated_dth": nominated,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_nomination_summary(state_data, nominations):
    """Compare each state's selected gas-day forecast with its nominations."""
    rows = []
    for state, (month_df, model) in state_data.items():
        baseline_daily = daily_forecast(model, month_df, np.zeros(len(month_df)))
        sub = nominations[nominations["state"] == state].reset_index(drop=True)
        nominated = float(sub["nominated_dth"].iloc[0])
        forecast = float(baseline_daily[0])
        rows.append(
            {
                "State": state,
                "Forecast (Dth/day)": forecast,
                "Actual nominations (Dth/day)": nominated,
                "Forecast minus actual nominations (Dth/day)": forecast - nominated,
                "Gap (%)": (forecast - nominated) / nominated * 100 if nominated else 0.0,
                "Nominations source": (
                    sub["source"].iloc[0] if "source" in sub else "Sample (generated)"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_sample_customer_breakdown(state_data, nominations):
    """Allocate each state's forecast and nominations to illustrative customers."""
    customer_shares = (0.35, 0.25, 0.20, 0.12, 0.08)
    rows = []
    for state, (gas_day_df, model) in state_data.items():
        gas_day = pd.Timestamp(gas_day_df["date"].iloc[0]).normalize()
        state_nominations = nominations[
            (nominations["state"] == state)
            & (pd.to_datetime(nominations["date"]).dt.normalize() == gas_day)
        ]
        if state_nominations.empty:
            raise ValueError(f"No nominations are available for {state} on {gas_day:%Y-%m-%d}.")

        forecast = float(
            daily_forecast(model, gas_day_df, np.zeros(len(gas_day_df)))[0]
        )
        nominated = float(state_nominations["nominated_dth"].iloc[0])

        def allocate(total):
            total = round(total, 1)
            amounts = [round(total * share, 1) for share in customer_shares]
            amounts[-1] = round(total - sum(amounts[:-1]), 1)
            return amounts

        customer_forecasts = allocate(forecast)
        customer_nominations = allocate(nominated)
        for index, (customer_forecast, customer_nomination) in enumerate(
            zip(customer_forecasts, customer_nominations), start=1
        ):
            rows.append(
                {
                    "State": state,
                    "Company / Scheduling Point": f"Customer {index}",
                    "Forecast (Dth/day)": customer_forecast,
                    "Actual nominations (Dth/day)": customer_nomination,
                    "Difference (Dth/day)": round(
                        customer_forecast - customer_nomination, 1
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_customer_nomination_guidance(
    history, state, forecast_dth, gas_day, history_days=90
):
    """Compare forecast-based customer targets with their recent nomination history."""
    required = {"date", "state", "customer", "nominated_dth"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(
            "Nomination history is missing required columns: "
            + ", ".join(sorted(missing))
        )

    target_date = pd.Timestamp(gas_day).normalize()
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    recent = history.loc[
        (history["state"] == state)
        & (history["date"] < target_date)
        & (history["date"] >= target_date - pd.Timedelta(days=history_days))
    ].copy()
    recent["nominated_dth"] = pd.to_numeric(
        recent["nominated_dth"], errors="coerce"
    )
    recent = recent.dropna(subset=["customer", "nominated_dth"])
    recent = recent.loc[recent["nominated_dth"] >= 0]
    if recent.empty:
        raise ValueError(
            f"No prior customer nomination history is available for {state} "
            f"within {history_days} days of {target_date:%Y-%m-%d}."
        )

    daily_totals = recent.groupby("date")["nominated_dth"].transform("sum")
    recent["share"] = np.divide(
        recent["nominated_dth"],
        daily_totals,
        out=np.zeros(len(recent), dtype=float),
        where=daily_totals.to_numpy() > 0,
    )
    stats = recent.groupby("customer").agg(
        history_days=("date", "nunique"),
        mean_share=("share", "mean"),
        median_nomination=("nominated_dth", "median"),
        p10_nomination=("nominated_dth", lambda values: values.quantile(0.10)),
        p90_nomination=("nominated_dth", lambda values: values.quantile(0.90)),
    )
    total_share = stats["mean_share"].sum()
    if total_share <= 0:
        raise ValueError(f"Nomination history for {state} has no positive daily totals.")
    stats["mean_share"] /= total_share
    stats["target_dth"] = (forecast_dth * stats["mean_share"]).round(1)
    stats["target_minus_median"] = (
        stats["target_dth"] - stats["median_nomination"]
    ).round(1)
    stats["variation_pct"] = np.where(
        stats["median_nomination"] > 0,
        (stats["p90_nomination"] - stats["p10_nomination"])
        / stats["median_nomination"] * 100,
        0.0,
    )

    def suggestion(row):
        target = row["target_dth"]
        if target > row["p90_nomination"]:
            return (
                f"Plan about {target:,.1f} Dth/day; this is above the recent "
                "nomination range. Confirm the demand increase and supply capacity."
            )
        if target < row["p10_nomination"]:
            return (
                f"Plan about {target:,.1f} Dth/day; this is below the recent "
                "nomination range. Confirm the expected demand reduction before "
                "lowering scheduled supply."
            )
        return (
            f"Plan about {target:,.1f} Dth/day; this is within the recent "
            "nomination range. Monitor the final forecast and nomination."
        )

    stats["suggestion"] = stats.apply(suggestion, axis=1)
    result = stats.reset_index().rename(
        columns={
            "customer": "Customer / Scheduling Point",
            "history_days": "History days",
            "median_nomination": "Historical median (Dth/day)",
            "p10_nomination": "Recent P10 (Dth/day)",
            "p90_nomination": "Recent P90 (Dth/day)",
            "variation_pct": "Historical variation (P10-P90, %)",
            "target_dth": "Forecast-based schedule target (Dth/day)",
            "target_minus_median": "Target minus historical median (Dth/day)",
            "suggestion": "Analyst suggestion",
        }
    )
    return result.drop(columns="mean_share")


def build_report(model, gas_day_df, extra_scenarios=()):
    """Calculate scenario impacts for the selected gas day.

    extra_scenarios: optional dicts with name, temp_shift, demand_mult.
    """
    base = daily_forecast(model, gas_day_df, np.zeros(len(gas_day_df))).sum()
    rows = []
    for s in list(rt.SCENARIOS) + list(extra_scenarios):
        frame = gas_day_df.copy()
        frame["temp_scenario"] = frame["temp_expected"] + s["temp_shift"]
        total = (
            model.predict(rt.make_features(frame, "temp_scenario"))
            * s["demand_mult"]
        ).sum()
        rows.append(
            {
                "scenario": s["name"],
                "forecast_dth": round(total, 1),
                "change_dth": round(total - base, 1),
                "change_pct": round((total - base) / base * 100, 1) if base else 0,
            }
        )
    report = pd.DataFrame(rows)
    order = report["change_pct"].abs().sort_values(ascending=False).index
    return report.loc[order].reset_index(drop=True)
