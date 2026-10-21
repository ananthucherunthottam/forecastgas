"""
Forecast red-team: trains a simple gas demand forecast, then attacks it with
stress scenarios to find where it is most likely to be wrong.

Run in the same folder as gas_data.csv (made by build_dataset.py):
    python red_team.py

Needs: pandas, numpy, scikit-learn  (all included in Anaconda)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---- settings you can change ----
DATA_FILE = "gas_data.csv"
TARGET_MONTH = str(pd.Timestamp.now().to_period("M") + 1)
REPORT_FILE = "risk_report.csv"
# ---------------------------------

# Each scenario: what could go differently from the plan?
# temp_shift = degrees F change, start/length = which days of the month (0 = day 1),
# demand_mult = change in demand not explained by weather (e.g. customer behaviour)
SCENARIOS = [
    dict(name="Cold snap (-10F for 5 days)", temp_shift=-10, start=9, length=5, demand_mult=1.0),
    dict(name="Colder month overall (-5F)", temp_shift=-5, start=0, length=999, demand_mult=1.0),
    dict(name="Warm spell (+8F for 7 days)", temp_shift=8, start=7, length=7, demand_mult=1.0),
    dict(name="Warmer month overall (+5F)", temp_shift=5, start=0, length=999, demand_mult=1.0),
    dict(name="Large customers use 8% less (assumed)", temp_shift=0, start=0, length=999, demand_mult=0.92),
    dict(name="Large customers use 8% more (assumed)", temp_shift=0, start=0, length=999, demand_mult=1.08),
    dict(name="Extended holiday slowdown (-15% for 4 days)", temp_shift=0, start=24, length=4, demand_mult=0.85),
]


def make_features(df, temp_col):
    """Turn a table of days into the numbers the model learns from."""
    return pd.DataFrame(
        {
            "heating_deg": np.clip(65 - df[temp_col], 0, None),  # how cold it is below 65F
            "is_weekend": (df["date"].dt.dayofweek >= 5).astype(int),
            "is_holiday": df["is_holiday"],
        }
    )


def expected_temperatures(history, dates):
    """What a forecaster would assume: the average temperature for each day of the year."""
    by_day = history.groupby(history["date"].dt.dayofyear)["temp_f"].mean()
    by_day = by_day.reindex(range(1, 367)).interpolate().bfill().ffill()
    smooth = by_day.rolling(7, center=True, min_periods=1).mean()
    return smooth.reindex(dates.dt.dayofyear).to_numpy()


def holiday_flags(dates):
    """Build holiday indicators for forecast dates not present in the dataset."""
    dates = pd.Series(pd.to_datetime(dates))
    md = dates.dt.strftime("%m-%d")
    holiday = md.isin(["12-24", "12-25", "12-26", "01-01", "07-04"])
    thanksgiving = (dates.dt.month == 11) & (dates.dt.day >= 22) & (
        dates.dt.dayofweek == 3
    ) & (dates.dt.day <= 28)
    return (holiday | thanksgiving).astype(int).to_numpy()


def prepare_month(df, month):
    """Split data into the month to forecast and the history before it, and train the model."""
    period = pd.Period(month)
    month_df = df[df["date"].dt.to_period("M") == period].copy()
    forecast_start = period.start_time
    history = df[df["date"] < forecast_start]
    if history.empty:
        raise ValueError(f"No training history is available before {period}.")
    if month_df.empty:
        forecast_dates = pd.date_range(period.start_time, period.end_time, freq="D")
        month_df = pd.DataFrame({"date": forecast_dates})
        month_df["is_holiday"] = holiday_flags(month_df["date"])
        month_df["usage"] = np.nan
    month_df["temp_expected"] = expected_temperatures(history, month_df["date"])
    model = LinearRegression().fit(make_features(history, "temp_f"), history["usage"])
    return month_df, model


def baseline_forecast(model, month_df):
    return model.predict(make_features(month_df, "temp_expected")).sum()


def apply_scenario(model, month_df, temp_shift=0, start=0, length=999, demand_mult=1.0, **_):
    """Total forecast for the month if the scenario happens instead of the plan."""
    temps = month_df["temp_expected"].to_numpy(dtype=float).copy()
    end = min(start + length, len(temps))
    temps[start:end] += temp_shift
    frame = month_df.copy()
    frame["temp_scenario"] = temps
    pred = model.predict(make_features(frame, "temp_scenario"))
    pred[start:end] *= demand_mult
    return pred.sum()


def what_if(model, month_df, temp_shift):
    """Handy for a slider later: whole month shifted by temp_shift degrees F."""
    base = baseline_forecast(model, month_df)
    new = apply_scenario(model, month_df, temp_shift=temp_shift)
    return round((new - base) / base * 100, 1)


def backtest(df, months):
    """How wrong would the forecast have been in past months?"""
    rows = []
    for m in months:
        month_df, model = prepare_month(df, m)
        forecast = baseline_forecast(model, month_df)
        actual = month_df["usage"].sum()
        rows.append(
            {
                "month": m,
                "forecast_dth": round(forecast, 0),
                "actual_dth": round(actual, 0),
                "error_pct": round((forecast - actual) / actual * 100, 1),
            }
        )
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    if "state" not in df:
        df["state"] = "Ohio"

    past_months = [f"2024-{m:02d}" for m in range(1, 13)]
    reports = []
    for state, state_df in df.groupby("state", sort=True):
        bt = backtest(state_df, past_months)
        print(f"=== {state} backtest: how far off was the plain forecast in 2024? ===")
        print(bt.to_string(index=False))
        worst = bt.loc[bt["error_pct"].abs().idxmax()]
        print(f"\nWorst month: {worst['month']} (off by {worst['error_pct']}%)\n")

        month_df, model = prepare_month(state_df, TARGET_MONTH)
        base = baseline_forecast(model, month_df)
        print(f"=== {state} red-team report for {TARGET_MONTH} ===")
        print(f"Baseline forecast: {base:,.0f} Dth\n")

        rows = []
        for scenario in SCENARIOS:
            total = apply_scenario(model, month_df, **scenario)
            rows.append(
                {
                    "state": state,
                    "scenario": scenario["name"],
                    "forecast_dth": round(total, 0),
                    "change_dth": round(total - base, 0),
                    "change_pct": round((total - base) / base * 100, 1),
                }
            )
        report = pd.DataFrame(rows)
        report["abs_pct"] = report["change_pct"].abs()
        report = report.sort_values("abs_pct", ascending=False).drop(columns="abs_pct")
        print(report.drop(columns="state").to_string(index=False))
        reports.append(report)

        top = report.iloc[0]
        print(
            f"\nBiggest risk: '{top['scenario']}' could move the monthly total by "
            f"{top['change_pct']}% ({top['change_dth']:,.0f} Dth)."
        )
        print(f"What if the month is 12F colder? {what_if(model, month_df, -12)}% change\n")

    pd.concat(reports, ignore_index=True).to_csv(REPORT_FILE, index=False)
    print(f"Report saved to {REPORT_FILE}")


if __name__ == "__main__":
    main()
