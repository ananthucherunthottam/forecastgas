"""
Extra features for the Gas Forecast app (no Streamlit in here):
live weather, uncertainty range, Monte Carlo, sensitivity, multi-day scenarios,
model comparison, Excel/PDF export, and email/Teams alerts.
"""
import io
import os
import smtplib
from email.message import EmailMessage

import numpy as np
import pandas as pd

import red_team as rt

# City used for each state's live weather forecast (state capitals)
STATE_COORDS = {
    "Ohio": (39.96, -82.99),
    "Virginia": (37.54, -77.44),
    "Pennsylvania": (40.27, -76.88),
}
# ------------------------------------------------------------ weather
def hdd(temp_f, base=65.0):
    """Heating degree days for a daily mean temperature in F."""
    return np.maximum(base - np.asarray(temp_f, dtype=float), 0.0)


def fetch_weather_forecast(state, days=16):
    """Daily mean temperature forecast (F) from Open-Meteo. Raises on failure."""
    import requests

    lat, lon = STATE_COORDS[state]
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_mean",
            "temperature_unit": "fahrenheit",
            "timezone": "auto",
            "forecast_days": days,
        },
        timeout=10,
    )
    response.raise_for_status()
    daily = response.json()["daily"]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "temp_f": daily["temperature_2m_mean"],
        }
    ).dropna()


def apply_live_weather(frame, weather):
    """Replace temp_expected with forecast temperatures where available.

    Returns (new_frame, number_of_days_replaced).
    """
    if weather is None or weather.empty:
        return frame, 0
    out = frame.copy()
    lookup = dict(zip(weather["date"].dt.normalize(), weather["temp_f"]))
    days = out["date"].dt.normalize()
    mask = days.isin(list(lookup))
    if mask.any():
        out.loc[mask, "temp_expected"] = days[mask].map(lookup).astype(float)
    return out, int(mask.sum())


# ------------------------------------------------------------ uncertainty
def error_band(backtest, forecast, z=1.28):
    """Approximate 80% range from the root-mean-square backtest error (%).

    Returns (low, high, sigma_pct).
    """
    errors = pd.to_numeric(backtest["error_pct"], errors="coerce").dropna()
    if errors.empty:
        return forecast, forecast, 0.0
    sigma = float(np.sqrt((errors ** 2).mean()))
    half = forecast * z * sigma / 100
    return forecast - half, forecast + half, sigma


def _repeat_first_row(frame, n):
    return frame.iloc[[0]].loc[lambda d: d.index.repeat(n)].reset_index(drop=True)


def monte_carlo(model, gas_day_df, n=1000, temp_sigma=6.0, demand_sigma_pct=3.0, seed=7):
    """Simulate demand (Dth/day) for the first row of gas_day_df.

    Temperature error ~ Normal(0, temp_sigma F); demand noise ~ Normal(1, sigma%).
    Both sigmas are assumptions, not fitted values.
    """
    rng = np.random.default_rng(seed)
    frame = _repeat_first_row(gas_day_df, n)
    frame["temp_mc"] = frame["temp_expected"] + rng.normal(0, temp_sigma, n)
    demand = model.predict(rt.make_features(frame, "temp_mc"))
    return demand * rng.normal(1.0, demand_sigma_pct / 100, n)


def sensitivity_table(model, gas_day_df, shifts=None):
    """Forecast for the first row across a range of temperature shifts."""
    shifts = np.arange(-30, 31, 2) if shifts is None else np.asarray(shifts)
    frame = _repeat_first_row(gas_day_df, len(shifts))
    frame["temp_s"] = frame["temp_expected"] + shifts
    return pd.DataFrame(
        {
            "Temperature shift (F)": shifts,
            "Forecast (Dth/day)": model.predict(rt.make_features(frame, "temp_s")),
        }
    )


def scenario_matrix(month_df, model, scenarios):
    """Percent change vs baseline for every day and scenario (long format)."""
    frame = month_df.copy()
    frame["temp_s"] = frame["temp_expected"]
    base = np.asarray(model.predict(rt.make_features(frame, "temp_s")), dtype=float)
    rows = []
    for s in scenarios:
        frame = month_df.copy()
        frame["temp_s"] = frame["temp_expected"] + s["temp_shift"]
        scen = np.asarray(model.predict(rt.make_features(frame, "temp_s"))) * s["demand_mult"]
        pct = np.where(base != 0, (scen - base) / np.where(base != 0, base, 1) * 100, 0.0)
        rows.append(
            pd.DataFrame(
                {
                    "Gas Day": month_df["date"].dt.strftime("%Y-%m-%d").to_numpy(),
                    "Scenario": s["name"],
                    "Change (%)": np.round(pct, 1),
                    "Change (Dth/day)": np.round(scen - base, 1),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------ model comparison
def compare_models(state_df, test_start="2024-01-01"):
    """Compare candidate models on a daily hold-out (independent of the app model).

    Trains on data before test_start and tests on data from test_start onward.
    """
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    df = state_df.sort_values("date").copy()
    temp_cols = [
        c for c in df.columns if "temp" in c.lower() and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not temp_cols or "usage" not in df:
        raise ValueError("Needs a numeric temperature column and a 'usage' column.")
    temp = df[temp_cols[0]]
    feats = pd.DataFrame(
        {
            "temp": temp,
            "hdd": hdd(temp),
            "dow": df["date"].dt.dayofweek,
            "month": df["date"].dt.month,
            "doy": df["date"].dt.dayofyear,
        }
    )
    if "is_holiday" in df:
        feats["holiday"] = df["is_holiday"].astype(int)
    valid = feats.notna().all(axis=1) & df["usage"].notna() & (df["usage"] > 0)
    feats, usage, dates = feats[valid], df.loc[valid, "usage"], df.loc[valid, "date"]
    train = dates < pd.Timestamp(test_start)
    test = ~train
    if train.sum() < 60 or test.sum() < 30:
        raise ValueError("Not enough data before and after the test start date.")

    models = {
        "Linear regression": LinearRegression(),
        "Random forest": RandomForestRegressor(n_estimators=150, random_state=0),
        "Gradient boosting": GradientBoostingRegressor(random_state=0),
    }
    rows = []
    for name, model in models.items():
        model.fit(feats[train], usage[train])
        pred = model.predict(feats[test])
        actual = usage[test].to_numpy()
        rows.append(
            {
                "Model": name,
                "MAPE (%)": round(float(np.mean(np.abs(pred - actual) / actual) * 100), 2),
                "RMSE (Dth/day)": round(float(np.sqrt(np.mean((pred - actual) ** 2))), 1),
            }
        )
    return pd.DataFrame(rows).sort_values("MAPE (%)").reset_index(drop=True)


# ------------------------------------------------------------ export
def build_excel(sheets):
    """sheets: dict of sheet name -> DataFrame. Returns .xlsx bytes."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    return buffer.getvalue()


def _latin1(text):
    return str(text).encode("latin-1", "replace").decode("latin-1")


def build_pdf(title, sections):
    """sections: list of (heading, text, monospace). Returns PDF bytes."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _latin1(title), new_x="LMARGIN", new_y="NEXT")
    for heading, body, mono in sections:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, _latin1(heading), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Courier" if mono else "Helvetica", "", 8 if mono else 10)
        pdf.multi_cell(0, 5, _latin1(body), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


# ------------------------------------------------------------ notifications
def notifications_configured():
    return bool(os.environ.get("TEAMS_WEBHOOK_URL")) or bool(
        os.environ.get("SMTP_HOST") and os.environ.get("EMAIL_TO")
    )


def send_teams(text):
    import requests

    response = requests.post(
        os.environ["TEAMS_WEBHOOK_URL"], json={"text": text}, timeout=15
    )
    response.raise_for_status()


def send_email(subject, body):
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = os.environ.get("EMAIL_FROM", os.environ.get("SMTP_USER", ""))
    message["To"] = os.environ["EMAIL_TO"]
    message.set_content(body)
    with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", "587")), timeout=20) as smtp:
        smtp.starttls()
        if os.environ.get("SMTP_USER"):
            smtp.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASSWORD", ""))
        smtp.send_message(message)


def notify(subject, text):
    """Send to every configured channel. Returns a list of result strings."""
    results = []
    if os.environ.get("TEAMS_WEBHOOK_URL"):
        try:
            send_teams(f"{subject}\n\n{text}")
            results.append("Teams: sent")
        except Exception as exc:
            results.append(f"Teams: failed ({exc})")
    if os.environ.get("SMTP_HOST") and os.environ.get("EMAIL_TO"):
        try:
            send_email(subject, text)
            results.append("Email: sent")
        except Exception as exc:
            results.append(f"Email: failed ({exc})")
    return results or ["No notification channel is configured."]
