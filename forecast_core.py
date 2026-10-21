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
            "Actual forecast (Dth/day)": base_daily.round(1),
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


def merge_uploaded_nominations(sample, uploaded):
    """Replace sample nominations with uploaded ones where dates and states match."""
    out = sample.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["source"] = "Sample (generated)"
    if uploaded is None or uploaded.empty:
        return out
    merged = out.merge(
        uploaded.rename(columns={"nominated_dth": "uploaded_dth"}),
        on=["date", "state"],
        how="left",
    )
    has_upload = merged["uploaded_dth"].notna()
    merged.loc[has_upload, "nominated_dth"] = merged.loc[has_upload, "uploaded_dth"]
    merged.loc[has_upload, "source"] = "Uploaded"
    return merged.drop(columns="uploaded_dth")


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
                "Actual forecast (Dth/day)": forecast,
                "Actual nominations (Dth/day)": nominated,
                "Forecast minus actual nominations (Dth/day)": forecast - nominated,
                "Gap (%)": (forecast - nominated) / nominated * 100 if nominated else 0.0,
                "Nominations source": (
                    sub["source"].iloc[0] if "source" in sub else "Sample (generated)"
                ),
            }
        )
    return pd.DataFrame(rows)


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
