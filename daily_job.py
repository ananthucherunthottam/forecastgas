"""
Daily job: builds tomorrow's forecast risks for all states, writes a briefing
(LLM, or the built-in template if no LLM is available), saves it to the
'warnings' folder and sends it to Teams / email if those are configured.

Run by hand to test:
    python daily_job.py

Schedule it with Windows Task Scheduler (see README notes in the chat).
Needs the same environment variables as the app (LLM_API_KEY, and optionally
TEAMS_WEBHOOK_URL, SMTP_HOST, EMAIL_TO, USE_LIVE_WEATHER=1).
"""
import truststore
truststore.inject_into_ssl()

import os
from pathlib import Path

import pandas as pd

import explain as ex
import extras as xt
import forecast_core as fc
import red_team as rt

STATES = ("Ohio", "Virginia", "Pennsylvania")
OUT_DIR = Path(__file__).resolve().parent / "warnings"


def main():
    df = pd.read_csv(rt.DATA_FILE, parse_dates=["date"])
    day = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    day_text = day.strftime("%Y-%m-%d")
    month = str(day.to_period("M"))
    use_weather = os.environ.get("USE_LIVE_WEATHER") == "1"

    reports = {}
    for name in STATES:
        month_df, model = fc.prepare_forecast_month(
            df[df["state"] == name].copy(), month
        )
        row = month_df[month_df["date"].dt.normalize() == day].copy()
        if row.empty:
            print(f"No forecast available for {name} on {day_text}; skipped.")
            continue
        row = row.reset_index(drop=True)
        if use_weather:
            try:
                row, _ = xt.apply_live_weather(row, xt.fetch_weather_forecast(name))
            except Exception as exc:
                print(f"Live weather failed for {name}: {exc}")
        reports[name] = fc.build_report(model, row)

    if not reports:
        print("Nothing to report.")
        return

    ex.TARGET_GAS_DAY = day_text
    text = ex.call_llm(ex.build_briefing_prompt(reports))
    source = "LLM"
    if text is None:
        text = ex.template_briefing(reports)
        source = "built-in template"
        if ex.LAST_ERROR:
            print(f"LLM not used: {ex.LAST_ERROR}")

    OUT_DIR.mkdir(exist_ok=True)
    out_file = OUT_DIR / f"warning_{day_text}.txt"
    out_file.write_text(text, encoding="utf-8")
    print(f"=== Briefing for {day_text} (written by: {source}) ===\n")
    print(text)
    print(f"\nSaved to {out_file}")

    if xt.notifications_configured():
        for line in xt.notify(f"Gas forecast briefing for {day_text}", text):
            print(line)


if __name__ == "__main__":
    main()
