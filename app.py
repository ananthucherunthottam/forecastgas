"""
Forecast Red-Team app (Streamlit).

Put these files in ONE folder:
    gas_data.csv, red_team.py, forecast_core.py, extras.py, explain.py, app.py

Run from that folder:
    streamlit run app.py

Needs: see requirements.txt
"""
import truststore
truststore.inject_into_ssl()

import hmac
import os
from contextlib import contextmanager

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import explain as ex
import extras as xt
import forecast_core as fc
import red_team as rt

st.set_page_config(page_title="Nisource Gas Forecast", layout="wide")


# ---------------- Optional password gate ----------------
def check_password():
    """Simple shared-password gate. Active only if APP_PASSWORD is set.
    This is a convenience lock, not real authentication or SSO."""
    required = os.environ.get("APP_PASSWORD")
    if not required or st.session_state.get("auth_ok"):
        return True
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered, required):
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()


@contextmanager
def safe(label):
    """Show an error for one section instead of breaking the whole page."""
    try:
        yield
    except Exception as exc:
        st.error(f"{label} could not run: {exc}")


# ---------------- Cached loaders ----------------
@st.cache_data
def load_data():
    return pd.read_csv(rt.DATA_FILE, parse_dates=["date"])


@st.cache_resource(show_spinner="Training forecast models...")
def cached_prepare(state_name, month, today_key):
    data = load_data()
    return fc.prepare_forecast_month(data[data["state"] == state_name].copy(), month)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_backtest(state_name):
    data = load_data()
    past = [f"2024-{m:02d}" for m in range(1, 13)]
    return rt.backtest(data[data["state"] == state_name].copy(), past)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_weather(state_name):
    return xt.fetch_weather_forecast(state_name)


df = load_data()
required_states = ("Ohio", "Virginia", "Pennsylvania")
if "state" not in df:
    st.error(
        "The loaded gas_data.csv contains only the original Ohio dataset. "
        "Run `python build_dataset.py` to download or generate data for Ohio, "
        "Virginia, and Pennsylvania, then reload this app."
    )
    st.stop()
missing_states = set(required_states) - set(df["state"].dropna().unique())
if missing_states:
    st.error(
        "The dataset is missing state data for "
        f"{', '.join(sorted(missing_states))}. Run `python build_dataset.py` "
        "and reload this app."
    )
    st.stop()

today = pd.Timestamp.now().normalize()
current_month = today.to_period("M")
target_month = pd.Period(rt.TARGET_MONTH)
gas_days = set(
    df.loc[df["date"] >= "2022-01-01", "date"].dt.strftime("%Y-%m-%d")
)
for forecast_period in (current_month, target_month):
    gas_days.update(
        pd.date_range(
            forecast_period.start_time, forecast_period.end_time, freq="D"
        ).strftime("%Y-%m-%d")
    )
gas_days = sorted(gas_days)

# ---------------- Callbacks (run before the page reruns) ----------------
if "custom_scenarios" not in st.session_state:
    st.session_state["custom_scenarios"] = []


def slider_key():
    return (
        f"temperature_shift_{st.session_state['sel_state']}_"
        f"{st.session_state['sel_gas_day']}"
    )


def add_custom_scenario():
    name = st.session_state.get("cs_name", "").strip() or "Custom scenario"
    if not name.lower().endswith("(assumed)"):
        name += " (assumed)"
    st.session_state["custom_scenarios"].append(
        {
            "name": name,
            "temp_shift": float(st.session_state.get("cs_temp", 0)),
            "demand_mult": 1 + float(st.session_state.get("cs_demand", 0)) / 100,
        }
    )


def clear_custom_scenarios():
    st.session_state["custom_scenarios"] = []


def save_current_scenario():
    name = st.session_state.get("save_name", "").strip()
    if name:
        xt.save_scenario(
            name,
            st.session_state.get(slider_key(), 0),
            st.session_state["custom_scenarios"],
        )


def load_saved_scenario():
    data = xt.load_saved_scenarios().get(st.session_state.get("saved_choice"))
    if data:
        st.session_state[slider_key()] = int(data["temp_shift"])
        st.session_state["custom_scenarios"] = data.get("custom_scenarios", [])


# ---------------- Sidebar controls ----------------
st.sidebar.header("What-if controls")
state = st.sidebar.selectbox("State", required_states, key="sel_state")
default_gas_day = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
if default_gas_day not in gas_days:
    default_gas_day = target_month.start_time.strftime("%Y-%m-%d")
gas_day = st.sidebar.date_input(
    "Gas Day to forecast",
    value=pd.Timestamp(default_gas_day).date(),
    min_value=pd.Timestamp(gas_days[0]).date(),
    max_value=pd.Timestamp(gas_days[-1]).date(),
    key="sel_gas_day",
)
selected_gas_day = pd.Timestamp(gas_day)
month = str(selected_gas_day.to_period("M"))
show_actual = st.sidebar.checkbox("Show what actually happened (hindsight)", value=False)
use_live_weather = st.sidebar.checkbox(
    "Use live weather forecast (Open-Meteo, next 16 days)",
    value=False,
    help="Replaces the historical-average temperature when the gas day is within "
    "the next 16 days. Needs internet access to open-meteo.com.",
)
alert_threshold = st.sidebar.number_input(
    "Alert when forecast differs from nominations by more than (%)",
    min_value=0.5,
    max_value=50.0,
    value=5.0,
    step=0.5,
)
st.sidebar.caption(
    "Temperatures use historical Open-Meteo data when available and synthetic data "
    "as a fallback. Gas usage is synthetic and illustrative, expressed in Dth/day. "
    "Customer-behaviour scenarios are assumptions. Sample nominations are generated "
    "for each state and day unless you upload real ones."
)

with st.sidebar.expander("Custom scenario"):
    st.text_input("Name", key="cs_name", placeholder="Cold snap plus growth")
    st.number_input("Temperature shift (F)", -30.0, 30.0, 0.0, 1.0, key="cs_temp")
    st.number_input("Demand change (%)", -50.0, 50.0, 0.0, 1.0, key="cs_demand")
    c_add, c_clear = st.columns(2)
    c_add.button("Add", on_click=add_custom_scenario)
    c_clear.button("Clear all", on_click=clear_custom_scenarios)
    for cs in st.session_state["custom_scenarios"]:
        st.caption(
            f"{cs['name']}: {cs['temp_shift']:+.0f} F, "
            f"{(cs['demand_mult'] - 1) * 100:+.0f}% demand"
        )

with st.sidebar.expander("Saved scenarios"):
    st.text_input("Save current settings as", key="save_name")
    st.button("Save", on_click=save_current_scenario)
    saved_names = list(xt.load_saved_scenarios())
    if saved_names:
        st.selectbox("Load", saved_names, key="saved_choice")
        st.button("Load selected", on_click=load_saved_scenario)
    else:
        st.caption("No saved scenarios yet.")

with st.sidebar.expander("Real nominations (upload)"):
    st.caption("CSV or Excel with columns: date, state, nominated_dth")
    nomination_file = st.file_uploader(
        "Nominations file", type=["csv", "xlsx"], label_visibility="collapsed"
    )

with st.sidebar.expander("LLM connection"):
    key_input = st.text_input("LLM API key (optional)", type="password")
    if key_input:
        os.environ["LLM_API_KEY"] = key_input
    st.caption(f"Model: {ex.MODEL}")
    st.caption("Key found." if os.environ.get("LLM_API_KEY") else "No key set.")

extra_scenarios = st.session_state["custom_scenarios"]

uploaded_nominations = None
if nomination_file is not None:
    with safe("Nominations upload"):
        uploaded_nominations = xt.parse_uploaded_nominations(
            nomination_file, nomination_file.name
        )

# ---------------- Build forecasts ----------------
try:
    monthly_state_data = {
        name: cached_prepare(name, month, str(today.date()))
        for name in required_states
    }
except Exception as exc:
    st.error(f"Could not build the forecast: {exc}")
    st.stop()

weather_by_state = {}
weather_note = None
if use_live_weather:
    try:
        weather_by_state = {name: cached_weather(name) for name in required_states}
    except Exception as exc:
        weather_note = f"Live weather unavailable, using historical averages: {exc}"

state_data = {}
live_days = {}
try:
    for name, (forecast_month, state_model) in monthly_state_data.items():
        gas_day_forecast = forecast_month[
            forecast_month["date"].dt.normalize() == selected_gas_day
        ].copy()
        if gas_day_forecast.empty:
            raise ValueError(f"No gas-day forecast is available for {gas_day} in {name}.")
        gas_day_forecast, live_days[name] = xt.apply_live_weather(
            gas_day_forecast.reset_index(drop=True), weather_by_state.get(name)
        )
        state_data[name] = (gas_day_forecast, state_model)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

sample_nominations = fc.build_sample_nominations(state_data)
all_nominations = fc.merge_uploaded_nominations(sample_nominations, uploaded_nominations)
nomination_summary = fc.build_nomination_summary(state_data, all_nominations)
month_df, model = state_data[state]
full_month_df = monthly_state_data[state][0].copy()

baseline_by_state = pd.DataFrame(
    [
        {
            "State": name,
            "Actual forecast (Dth/day)": rt.baseline_forecast(state_model, state_month),
        }
        for name, (state_month, state_model) in state_data.items()
    ]
)
daily_baseline_by_state = pd.concat(
    [
        pd.DataFrame(
            {
                "Gas Day": state_month["date"].dt.strftime("%Y-%m-%d"),
                "State": name,
                "Forecasted gas demand (Dth/day)": fc.daily_forecast(
                    state_model, state_month, np.zeros(len(state_month))
                ).round(1),
                "Forecasted temperature (F)": state_month["temp_expected"].round(1),
            }
        )
        for name, (state_month, state_model) in state_data.items()
    ],
    ignore_index=True,
)
training_cutoff = df.loc[
    df["date"] < selected_gas_day.to_period("M").start_time, "date"
].max()

st.sidebar.subheader(f"Temperature controls, {gas_day}")
st.sidebar.caption(
    "Apply a temperature shift to this gas day. A zero shift leaves its expected "
    "temperature unchanged."
)
temp_key = f"temperature_shift_{state}_{gas_day}"
st.session_state.setdefault(temp_key, 0)
temp_shift = st.sidebar.slider(
    "Temperature change on selected gas day (F)",
    min_value=-30,
    max_value=30,
    step=1,
    key=temp_key,
)
temp_shifts = np.full(len(month_df), temp_shift)
st.sidebar.caption(f"Applying {temp_shift:+d} F to gas day {gas_day}.")

base_daily = fc.daily_forecast(model, month_df, np.zeros(len(month_df)))
whatif_daily = fc.daily_forecast(model, month_df, temp_shifts)
base_total = float(base_daily.sum())
whatif_total = float(whatif_daily.sum())

# Approximate range from past forecast error
band_low = band_high = band_sigma = None
try:
    band_low, band_high, band_sigma = xt.error_band(cached_backtest(state), base_total)
except Exception:
    pass

state_nominations = all_nominations[all_nominations["state"] == state][
    ["date", "nominated_dth", "source"]
]
nominated_daily = state_nominations["nominated_dth"].to_numpy()
forecast_minus_nominated = base_daily - nominated_daily

daily_table = fc.daily_forecast_table(month_df, temp_shifts, base_daily, whatif_daily)
daily_table["Heating degree days"] = xt.hdd(
    month_df["temp_expected"].to_numpy() + temp_shifts
).round(1)
daily_table["Gas Day"] = pd.to_datetime(daily_table["Gas Day"])
daily_table = daily_table.merge(
    state_nominations, left_on="Gas Day", right_on="date", how="left"
).drop(columns="date")
daily_table["Gas Day"] = daily_table["Gas Day"].dt.strftime("%Y-%m-%d")
daily_table["Forecast minus nominated (Dth/day)"] = (
    daily_table["What-if forecast (Dth/day)"] - daily_table["nominated_dth"]
).round(1)
daily_table = daily_table.rename(
    columns={
        "nominated_dth": "Actual nominations (Dth/day)",
        "source": "Nominations source",
        "Forecast minus nominated (Dth/day)": "Forecast minus actual nominations (Dth/day)",
    }
)

report = fc.build_report(model, month_df, extra_scenarios)

# ---------------- Header and alerts ----------------
st.title("Nisource Gas Forecast")
st.caption("An AI critic that tries to break the demand forecast before it goes to suppliers.")
st.caption(f"Model training data available through {training_cutoff:%Y-%m-%d}.")
if weather_note:
    st.warning(weather_note)
if use_live_weather and not weather_note and live_days.get(state, 0) == 0:
    st.info(
        "Live weather covers the next 16 days only. This gas day uses the "
        "historical-average temperature."
    )

for _, row in nomination_summary.iterrows():
    if abs(row["Gap (%)"]) > alert_threshold:
        st.error(
            f"ALERT {row['State']}: forecast is {row['Gap (%)']:+.1f}% "
            f"({row['Forecast minus actual nominations (Dth/day)']:+,.1f} Dth/day) "
            f"versus nominations for {gas_day}."
        )

tab_forecast, tab_risk, tab_sim, tab_track, tab_chat, tab_export = st.tabs(
    [
        "Forecast",
        "Risks and warnings",
        "Scenarios and simulation",
        "Tracking and data",
        "Ask the forecast",
        "Export",
    ]
)

# ======================= Tab 1: Forecast =======================
with tab_forecast:
    with safe("State map"):
        map_rows = []
        for _, row in nomination_summary.iterrows():
            gap = abs(row["Gap (%)"])
            if gap >= alert_threshold:
                color, status = "#c62828", "Alert"
            elif gap >= alert_threshold / 2:
                color, status = "#f9a825", "Watch"
            else:
                color, status = "#2e7d32", "OK"
            lat, lon = xt.STATE_COORDS[row["State"]]
            map_rows.append(
                {"State": row["State"], "lat": lat, "lon": lon, "color": color,
                 "size": 40000, "Status": status, "Gap (%)": round(row["Gap (%)"], 1)}
            )
        map_df = pd.DataFrame(map_rows)
        map_col, legend_col = st.columns([2, 1])
        with map_col:
            st.map(map_df, latitude="lat", longitude="lon", color="color", size="size")
        with legend_col:
            st.markdown("**Forecast vs nominations**")
            st.dataframe(
                map_df[["State", "Status", "Gap (%)"]], hide_index=True,
                use_container_width=True,
            )
            st.caption("Green: small gap. Amber: over half the alert threshold. Red: over it.")

    st.subheader(f"Actual forecast by state, Gas Day {gas_day}")
    st.dataframe(
        baseline_by_state.style.format({"Actual forecast (Dth/day)": "{:,.1f}"}),
        hide_index=True,
        use_container_width=True,
    )
    st.subheader(f"Gas-day actual forecast, {state}, {gas_day}")
    st.caption("Temperature estimates use historical averages unless live weather is on.")
    st.dataframe(
        daily_baseline_by_state[
            (daily_baseline_by_state["State"] == state)
            & (daily_baseline_by_state["Gas Day"] == gas_day)
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.subheader(f"Actual nominations by state, Gas Day {gas_day}")
    st.caption(
        "Sample nominations are generated from each state's forecast with repeatable "
        "variation of up to 4%. Rows marked Uploaded come from your file."
    )
    st.dataframe(
        nomination_summary.style.format(
            {
                "Actual forecast (Dth/day)": "{:,.1f}",
                "Actual nominations (Dth/day)": "{:,.1f}",
                "Forecast minus actual nominations (Dth/day)": "{:+,.1f}",
                "Gap (%)": "{:+.1f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{state} actual forecast for {gas_day}", f"{base_total:,.1f} Dth")
    c2.metric(
        "What-if forecast",
        f"{whatif_total:,.1f} Dth",
        delta=f"{(whatif_total - base_total) / base_total * 100:+.1f}%" if base_total else None,
    )
    c3.metric("Change from baseline", f"{whatif_total - base_total:+,.1f} Dth")
    if band_low is not None:
        st.caption(
            f"Approximate 80% range for the actual forecast: {band_low:,.1f} to "
            f"{band_high:,.1f} Dth. Based on the 2024 monthly backtest error "
            f"(about {band_sigma:.1f}%); daily error is usually larger, so treat it as a guide."
        )

    st.subheader(f"Forecast vs actual nominations, {state}, Gas Day {gas_day}")
    n1, n2, n3 = st.columns(3)
    n1.metric("Actual nominations", f"{nominated_daily[0]:,.1f} Dth")
    n2.metric("Actual forecast", f"{base_daily[0]:,.1f} Dth")
    n3.metric(
        "Forecast minus actual nominations", f"{forecast_minus_nominated[0]:+,.1f} Dth"
    )

    bars = [
        {"Series": "Actual forecast", "Dth/day": float(base_daily[0])},
        {"Series": "What-if forecast", "Dth/day": float(whatif_daily[0])},
        {"Series": "Nominations", "Dth/day": float(nominated_daily[0])},
    ]
    if show_actual and pd.notna(month_df["usage"].iloc[0]):
        bars.append(
            {"Series": "Actual usage (hindsight)", "Dth/day": float(month_df["usage"].iloc[0])}
        )
    bar_chart = (
        alt.Chart(pd.DataFrame(bars))
        .mark_bar()
        .encode(
            x=alt.X("Series:N", sort=None, title=None),
            y=alt.Y("Dth/day:Q"),
            color=alt.Color("Series:N", legend=None),
            tooltip=["Series", alt.Tooltip("Dth/day:Q", format=",.1f")],
        )
    )
    if band_low is not None:
        range_df = pd.DataFrame(
            [{"Series": "Actual forecast", "low": band_low, "high": band_high}]
        )
        bar_chart = bar_chart + alt.Chart(range_df).mark_rule(size=3, color="black").encode(
            x=alt.X("Series:N", sort=None), y="low:Q", y2="high:Q"
        )
    st.altair_chart(bar_chart, use_container_width=True)
    st.caption(
        "A positive gap means the forecast is above the nominations. "
        "Values are illustrative because gas usage is synthetic. "
        "The local usage dataset may end before later gas days; those are forecast-only."
    )
    st.dataframe(daily_table, hide_index=True, use_container_width=True)
    st.download_button(
        "Download gas-day forecast (CSV)",
        data=daily_table.to_csv(index=False).encode("utf-8"),
        file_name=f"gas_day_forecast_{state.lower()}_{gas_day}.csv",
        mime="text/csv",
    )

# ======================= Tab 2: Risks and warnings =======================
allowed_extra = [
    base_total, whatif_total, temp_shift, float(nominated_daily[0]),
    *[s["temp_shift"] for s in rt.SCENARIOS],
    *[s["temp_shift"] for s in extra_scenarios],
]

with tab_risk:
    st.subheader(f"Risks to the {gas_day} forecast, {state}")
    st.caption(
        "Scenario impacts apply their assumed temperature and demand changes to "
        "this gas day only. Custom scenarios from the sidebar are included."
    )
    st.dataframe(
        report.rename(
            columns={"forecast_dth": "Forecast (Dth/day)", "change_dth": "Change (Dth/day)"}
        ),
        hide_index=True,
        use_container_width=True,
    )

    tone = st.radio(
        "Warning style", list(ex.TONES), index=list(ex.TONES).index(ex.DEFAULT_TONE),
        horizontal=True,
    )

    st.subheader("Analyst warning")
    if st.button("Generate warning"):
        ex.TARGET_GAS_DAY = gas_day
        with st.spinner("Asking the LLM..."):
            text = ex.call_llm(ex.build_prompt(report, state=state, tone=tone))
        source, error = "LLM", None
        if text is None:
            text = ex.template_warning(report, state=state)
            source, error = "built-in template", ex.LAST_ERROR
        st.session_state["warning"] = (text, source, gas_day, state, error)
    saved = st.session_state.get("warning")
    if saved and saved[2:4] == (gas_day, state):
        st.write(saved[0])
        st.caption(f"Written by: {saved[1]}")
        if saved[4]:
            st.warning(f"LLM not used: {saved[4]}")
        if saved[1] == "LLM":
            unsupported = ex.check_numbers(saved[0], report, allowed_extra)
            if unsupported:
                st.warning(
                    "Fact check: these numbers were not found in the data, please verify: "
                    + ", ".join(unsupported)
                )
            else:
                st.success("Fact check: every number in this warning matches the data.")
        if xt.notifications_configured():
            if st.button("Send this warning to Teams / email"):
                for line in xt.notify(f"Gas forecast warning: {state}, {gas_day}", saved[0]):
                    st.caption(line)
    else:
        st.info("Press the button to generate a plain-English warning for this gas day.")

    st.subheader("Briefing for all states")
    if st.button("Generate briefing for all states"):
        ex.TARGET_GAS_DAY = gas_day
        reports = {
            name: fc.build_report(m, d, extra_scenarios)
            for name, (d, m) in state_data.items()
        }
        with st.spinner("Asking the LLM..."):
            text = ex.call_llm(ex.build_briefing_prompt(reports, tone))
        source, error = "LLM", None
        if text is None:
            text = ex.template_briefing(reports)
            source, error = "built-in template", ex.LAST_ERROR
        st.session_state["briefing"] = (
            text, source, gas_day, error, pd.concat(reports.values(), ignore_index=True)
        )
    brief = st.session_state.get("briefing")
    if brief and brief[2] == gas_day:
        st.write(brief[0])
        st.caption(f"Written by: {brief[1]}")
        if brief[3]:
            st.warning(f"LLM not used: {brief[3]}")
        if brief[1] == "LLM":
            unsupported = ex.check_numbers(brief[0], brief[4], allowed_extra)
            if unsupported:
                st.warning(
                    "Fact check: these numbers were not found in the data, please verify: "
                    + ", ".join(unsupported)
                )
            else:
                st.success("Fact check: every number in this briefing matches the data.")
    if not xt.notifications_configured():
        st.caption(
            "To send warnings by Teams or email, set TEAMS_WEBHOOK_URL, or SMTP_HOST "
            "and EMAIL_TO (plus SMTP_PORT, SMTP_USER, SMTP_PASSWORD) before starting the app."
        )

# ======================= Tab 3: Scenarios and simulation =======================
with tab_sim:
    st.subheader(f"Sensitivity to temperature, {state}, {gas_day}")
    with safe("Sensitivity chart"):
        sens = xt.sensitivity_table(model, month_df)
        line = alt.Chart(sens).mark_line().encode(
            x="Temperature shift (F):Q",
            y=alt.Y("Forecast (Dth/day):Q", scale=alt.Scale(zero=False)),
            tooltip=[
                "Temperature shift (F)",
                alt.Tooltip("Forecast (Dth/day):Q", format=",.1f"),
            ],
        )
        nom_rule = alt.Chart(pd.DataFrame({"y": [float(nominated_daily[0])]})).mark_rule(
            strokeDash=[5, 5], color="red"
        ).encode(y="y:Q")
        st.altair_chart(line + nom_rule, use_container_width=True)
        st.caption("The dashed red line is the nominations. Where the curve crosses it, demand equals nominations.")

    st.subheader("Weather risk simulation (Monte Carlo)")
    with safe("Monte Carlo simulation"):
        m1, m2, m3 = st.columns(3)
        runs = m1.slider("Simulated days", 200, 5000, 1000, step=100)
        temp_sigma = m2.slider("Temperature error, 1 std dev (F)", 1.0, 15.0, 6.0, 0.5)
        demand_sigma = m3.slider("Other demand noise, 1 std dev (%)", 0.0, 10.0, 3.0, 0.5)
        sims = xt.monte_carlo(model, month_df, runs, temp_sigma, demand_sigma)
        nom = float(nominated_daily[0])
        p_over = float((sims > nom).mean() * 100)
        p_over5 = float((sims > nom * 1.05).mean() * 100)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Chance demand exceeds nominations", f"{p_over:.0f}%")
        s2.metric("Chance it exceeds by 5%+", f"{p_over5:.0f}%")
        s3.metric("Average simulated demand", f"{sims.mean():,.1f} Dth")
        s4.metric("95th percentile", f"{np.percentile(sims, 95):,.1f} Dth")
        hist = alt.Chart(pd.DataFrame({"Demand (Dth/day)": sims})).mark_bar().encode(
            x=alt.X("Demand (Dth/day):Q", bin=alt.Bin(maxbins=40)),
            y=alt.Y("count():Q", title="Simulated days"),
        )
        rule = alt.Chart(pd.DataFrame({"x": [nom]})).mark_rule(color="red", size=2).encode(x="x:Q")
        st.altair_chart(hist + rule, use_container_width=True)
        st.caption(
            "The temperature and noise spreads are assumptions you can change, not fitted values. "
            "The red line is the nominations."
        )

    st.subheader(f"Scenarios across the month, {state}")
    with safe("Multi-day scenario view"):
        month_dates = full_month_df["date"].dt.strftime("%Y-%m-%d").tolist()
        start, end = st.select_slider(
            "Date range", options=month_dates, value=(month_dates[0], month_dates[-1])
        )
        window = full_month_df[
            (full_month_df["date"] >= pd.Timestamp(start))
            & (full_month_df["date"] <= pd.Timestamp(end))
        ]
        matrix = xt.scenario_matrix(window, model, list(rt.SCENARIOS) + list(extra_scenarios))
        heat = alt.Chart(matrix).mark_rect().encode(
            x=alt.X("Gas Day:O"),
            y=alt.Y("Scenario:N", title=None),
            color=alt.Color("Change (%):Q", scale=alt.Scale(scheme="redyellowblue", reverse=True)),
            tooltip=["Gas Day", "Scenario", "Change (%)", "Change (Dth/day)"],
        )
        st.altair_chart(heat, use_container_width=True)
        st.caption("Uses historical-average temperatures for every day in the range.")

    st.subheader("Model comparison")
    with safe("Model comparison"):
        st.caption(
            "Trains three candidate models on data before 2024 and tests them on 2024 days. "
            "This is independent of the app's own model and does not change the forecast."
        )
        if st.button("Run model comparison"):
            with st.spinner("Training candidate models..."):
                st.session_state["model_compare"] = (
                    state, xt.compare_models(df[df["state"] == state].copy())
                )
        result = st.session_state.get("model_compare")
        if result and result[0] == state:
            st.dataframe(result[1], hide_index=True, use_container_width=True)
            st.caption("Lower MAPE and RMSE are better.")

# ======================= Tab 4: Tracking and data =======================
with tab_track:
    st.subheader(f"Forecast vs actual usage, {state}, {month}")
    with safe("Forecast tracking"):
        base_full = fc.daily_forecast(model, full_month_df, np.zeros(len(full_month_df)))
        track = pd.DataFrame(
            {
                "Gas Day": full_month_df["date"].to_numpy(),
                "Forecast": base_full,
                "Actual usage": full_month_df["usage"].to_numpy(),
            }
        ).dropna()
        if track.empty:
            st.info("No observed usage yet for this month, so there is nothing to compare.")
        else:
            track["Error (%)"] = (
                (track["Forecast"] - track["Actual usage"])
                / track["Actual usage"].replace(0, np.nan) * 100
            )
            t1, t2 = st.columns(2)
            t1.metric("Average absolute error", f"{track['Error (%)'].abs().mean():.1f}%")
            t2.metric("Days compared", f"{len(track)}")
            long = track.melt("Gas Day", ["Forecast", "Actual usage"], "Series", "Dth/day")
            st.altair_chart(
                alt.Chart(long).mark_line(point=True).encode(
                    x="Gas Day:T", y=alt.Y("Dth/day:Q", scale=alt.Scale(zero=False)),
                    color="Series:N",
                    tooltip=["Gas Day:T", "Series", alt.Tooltip("Dth/day:Q", format=",.1f")],
                ),
                use_container_width=True,
            )
            st.altair_chart(
                alt.Chart(track).mark_bar().encode(
                    x="Gas Day:T", y="Error (%):Q",
                    tooltip=["Gas Day:T", alt.Tooltip("Error (%):Q", format=".1f")],
                ),
                use_container_width=True,
            )

    st.subheader("Data and connection status")
    status_rows = []
    for name in required_states:
        sub = df[df["state"] == name]
        status_rows.append(
            {
                "State": name,
                "Rows": len(sub),
                "First date": sub["date"].min().strftime("%Y-%m-%d"),
                "Latest date": sub["date"].max().strftime("%Y-%m-%d"),
                "Live weather days used": live_days.get(name, 0),
            }
        )
    st.dataframe(pd.DataFrame(status_rows), hide_index=True, use_container_width=True)
    st.caption(
        f"Live weather: {'on' if use_live_weather else 'off'}. "
        f"LLM key: {'found' if os.environ.get('LLM_API_KEY') else 'not set'}. "
        f"LLM model: {ex.MODEL}. Nominations: "
        f"{'uploaded file' if uploaded_nominations is not None else 'generated samples'}."
    )

    with st.expander("Proof: how wrong was the plain forecast in 2024?"):
        with safe("Backtest"):
            bt = cached_backtest(state)
            st.bar_chart(bt.set_index("month")["error_pct"])
            st.dataframe(bt, hide_index=True)

# ======================= Tab 5: Ask the forecast =======================
with tab_chat:
    st.subheader("Ask the forecast")
    st.caption(
        "Questions are answered only from the tables on this page. Try: "
        "'Why is the biggest risk bigger than the others?' or 'Which state has the largest gap?'"
    )
    context = "\n".join(
        [
            f"Selected state: {state}. Gas day: {gas_day}. Temperature shift applied: {temp_shift:+d} F.",
            "",
            "Forecast by state (Dth/day):",
            baseline_by_state.round(1).to_csv(index=False),
            "Forecast versus nominations:",
            nomination_summary.round(1).to_csv(index=False),
            f"Scenario risks for {state}:",
            ex.report_as_text(report),
            "",
            "Gas-day details:",
            daily_baseline_by_state[
                (daily_baseline_by_state["Gas Day"] == gas_day)
            ].to_csv(index=False),
        ]
    )
    chat_key = f"chat_{state}_{gas_day}"
    history = st.session_state.setdefault(chat_key, [])
    for role, content in history:
        with st.chat_message(role):
            st.write(content)
    question = st.chat_input(
        f"Ask about the {gas_day} forecast",
        key=f"chat_input_{state}_{gas_day}",
    )
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.spinner("Thinking..."):
            answer = ex.ask_llm(question, context, history)
        if answer is None:
            answer = f"The LLM is not available: {ex.LAST_ERROR}"
        with st.chat_message("assistant"):
            st.write(answer)
        history.append(("user", question))
        history.append(("assistant", answer))
    if history and st.button("Clear conversation", key=f"clear_{state}_{gas_day}"):
        st.session_state[chat_key] = []
        st.rerun()

# ======================= Tab 6: Export =======================
with tab_export:
    st.subheader("Export")
    warning_text = ""
    saved = st.session_state.get("warning")
    if saved and saved[2:4] == (gas_day, state):
        warning_text = saved[0]
    brief = st.session_state.get("briefing")
    briefing_text = brief[0] if brief and brief[2] == gas_day else ""

    with safe("Excel export"):
        sheets = {
            "Forecast by state": baseline_by_state,
            "Nominations": nomination_summary,
            "Daily detail": daily_table,
            "Risks": report,
        }
        if warning_text:
            sheets["Warning"] = pd.DataFrame({"Warning": [warning_text]})
        if briefing_text:
            sheets["Briefing"] = pd.DataFrame({"Briefing": [briefing_text]})
        st.download_button(
            "Download Excel workbook",
            data=xt.build_excel(sheets),
            file_name=f"gas_forecast_{state.lower()}_{gas_day}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with safe("PDF export"):
        sections = [
            ("Forecast by state (Dth/day)", baseline_by_state.round(1).to_string(index=False), True),
            ("Forecast versus nominations", nomination_summary.round(1).to_string(index=False), True),
            (f"Scenario risks, {state}", report.to_string(index=False), True),
        ]
        if warning_text:
            sections.append((f"Analyst warning, {state}", warning_text, False))
        if briefing_text:
            sections.append(("Briefing for all states", briefing_text, False))
        sections.append(
            ("Notes", "Gas usage and sample nominations are synthetic and illustrative. "
             "Scenarios marked 'assumed' are assumptions, not measured facts.", False)
        )
        st.download_button(
            "Download PDF report",
            data=xt.build_pdf(f"Gas Forecast Report, {state}, {gas_day}", sections),
            file_name=f"gas_forecast_{state.lower()}_{gas_day}.pdf",
            mime="application/pdf",
        )
    st.caption("Generate the warning or briefing first if you want them included.")
