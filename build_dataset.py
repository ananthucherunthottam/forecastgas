"""
Builds gas_data.csv and customer_nomination_history.csv. Gas usage and customer
nominations are illustrative synthetic data; only downloaded temperatures may be real.

Run on a machine with internet:  python build_dataset.py
If the download fails (e.g. blocked network), it falls back to synthetic
temperatures so you still get a usable file.

Needs: pandas, numpy
"""
import json
import urllib.request

import numpy as np
import pandas as pd

# ---- settings you can change ----
LOCATIONS = {
    "Ohio": (39.9612, -82.9988, 0),          # Columbus
    "Virginia": (37.5407, -77.4360, 3),      # Richmond
    "Pennsylvania": (40.2732, -76.8867, -2), # Harrisburg
}
START = "2021-01-01"
END = min(
    pd.Timestamp("2026-10-31"),
    pd.Timestamp.now().normalize() - pd.Timedelta(days=1),
).strftime("%Y-%m-%d")
OUT_FILE = "gas_data.csv"
NOMINATION_OUT_FILE = "customer_nomination_history.csv"
SEED = 42
CUSTOMER_SHARES = (0.35, 0.25, 0.20, 0.12, 0.08)
CUSTOMER_VOLATILITY = (0.03, 0.05, 0.08, 0.10, 0.14)
# ---------------------------------


def fetch_temperatures(latitude, longitude):
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={latitude}&longitude={longitude}"
        f"&start_date={START}&end_date={END}"
        "&daily=temperature_2m_mean"
        "&temperature_unit=fahrenheit&timezone=auto"
    )
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(data["daily"]["time"]),
            "temp_f": data["daily"]["temperature_2m_mean"],
        }
    )
    return df.dropna()


def synthetic_temperatures(seed=SEED, temperature_offset=0):
    rng = np.random.default_rng(seed)
    last_completed_day = min(
        pd.Timestamp(END), pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    )
    days = pd.date_range(START, last_completed_day)
    temp = (
        52 + temperature_offset
        + 28 * np.sin((days.dayofyear - 105) / 365.25 * 2 * np.pi)
        + rng.normal(0, 6, len(days))
    )
    return pd.DataFrame({"date": days, "temp_f": temp.round(1)})


def add_usage(df, seed=SEED):
    """Generate illustrative synthetic daily usage values expressed in Dth/day."""
    rng = np.random.default_rng(seed)

    # Heating demand grows as temperature drops below 65F
    heating = np.clip(65 - df["temp_f"], 0, None) * 2.8
    base = 40  # illustrative Dth/day for cooking, hot water, etc.
    usage = base + heating

    # Weekends use a bit less (offices and industry slow down)
    usage = usage * np.where(df["date"].dt.dayofweek >= 5, 0.92, 1.0)

    # Rough holiday dip (Dec 24-26, Jan 1, Jul 4, Thanksgiving week)
    md = df["date"].dt.strftime("%m-%d")
    holiday = md.isin(["12-24", "12-25", "12-26", "01-01", "07-04"])
    thanksgiving = (df["date"].dt.month == 11) & (df["date"].dt.day >= 22) & (
        df["date"].dt.dayofweek == 3
    ) & (df["date"].dt.day <= 28)
    df["is_holiday"] = (holiday | thanksgiving).astype(int)
    usage = usage * np.where(df["is_holiday"] == 1, 0.85, 1.0)

    # Random noise
    usage = usage + rng.normal(0, 8, len(df))
    df["usage"] = np.clip(usage, 10, None).round(1)
    return df


def build_customer_nomination_history(usage_data, seed=SEED):
    """Create illustrative customer nominations from the synthetic usage series."""
    required = {"date", "state", "usage"}
    missing = required - set(usage_data.columns)
    if missing:
        raise ValueError(
            "Usage data is missing required columns: " + ", ".join(sorted(missing))
        )

    frames = []
    for state_index, (state, state_data) in enumerate(
        usage_data.groupby("state", sort=True)
    ):
        state_data = state_data.sort_values("date").reset_index(drop=True)
        rng = np.random.default_rng(seed + state_index)
        season = 0.025 * np.sin(
            2 * np.pi * state_data["date"].dt.dayofyear.to_numpy() / 365.25
        )
        for customer_index, (share, volatility) in enumerate(
            zip(CUSTOMER_SHARES, CUSTOMER_VOLATILITY), start=1
        ):
            customer_bias = (customer_index - 3) * 0.015
            variation = rng.normal(0, volatility, len(state_data))
            nominated = (
                state_data["usage"].to_numpy()
                * share
                * np.maximum(0.5, 1 + customer_bias + season + variation)
            )
            frames.append(
                pd.DataFrame(
                    {
                        "date": state_data["date"].to_numpy(),
                        "state": state,
                        "customer": f"Sample Customer {customer_index}",
                        "nominated_dth": nominated.round(1),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def main():
    state_data = []
    for index, (state, (latitude, longitude, temperature_offset)) in enumerate(
        LOCATIONS.items()
    ):
        try:
            state_df = fetch_temperatures(latitude, longitude)
            print(f"Downloaded {len(state_df)} days of real temperatures for {state}.")
        except Exception as exc:
            print(
                f"Could not download temperatures for {state} ({exc}). "
                "Using synthetic temperatures."
            )
            state_df = synthetic_temperatures(
                seed=SEED + index, temperature_offset=temperature_offset
            )

        state_df["state"] = state
        state_data.append(add_usage(state_df, seed=SEED + index))

    df = pd.concat(state_data, ignore_index=True)
    df = df[["date", "state", "temp_f", "is_holiday", "usage"]]
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(df)} rows for {len(LOCATIONS)} states to {OUT_FILE}")
    nominations = build_customer_nomination_history(df)
    nominations.to_csv(NOMINATION_OUT_FILE, index=False)
    print(
        f"Saved {len(nominations)} illustrative rows to {NOMINATION_OUT_FILE}. "
        "These are synthetic, not historical customer nominations."
    )
    print(df.head())
    print("Correlation between temp and usage:", round(df["temp_f"].corr(df["usage"]), 3))


if __name__ == "__main__":
    main()
