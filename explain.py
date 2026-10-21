"""
LLM helpers for the Gas Forecast app.

- Turns a risk report into a short plain-English warning (3 tones).
- Writes one combined briefing for all states.
- Answers analyst questions using only the data the app shows.
- Fact-checks the numbers in an LLM answer against the report.

Run in the same folder to test:
    python explain.py

Works without an LLM (built-in template). For LLM wording set LLM_API_KEY.

Needs: pandas, truststore, langchain-openai
"""
# Use the Windows certificate store (fixes corporate SSL errors)
import truststore
truststore.inject_into_ssl()

import os
import re

import pandas as pd

# ---- settings you can change ----
REPORT_FILE = "risk_report.csv"
OUT_FILE = "warning.txt"
TARGET_GAS_DAY = "2024-01-01"
MODEL = os.environ.get("LLM_MODEL", "azure/genailab-maas-gpt-4o-mini")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://genailab.tcs.in")
# Set LLM_VERIFY_SSL=0 only if you still get certificate errors (hackathon use only)
VERIFY_SSL = os.environ.get("LLM_VERIFY_SSL", "1") != "0"
# ---------------------------------

# Holds the reason the last LLM call failed (shown in the Streamlit app)
LAST_ERROR = None

DEFAULT_TONE = "Executive summary"
TONES = {
    "Short alert": "under 60 words, one or two sentences: the biggest risk and one action",
    "Executive summary": "under 120 words",
    "Detailed analyst note": "under 250 words, covering the top three risks",
}


# ---------------------------------------------------------------- helpers
def _change_column(report):
    return "change_dth" if "change_dth" in report else "change_units"


def report_as_text(report):
    change_column = _change_column(report)
    lines = []
    for _, r in report.iterrows():
        lines.append(
            f"- {r['scenario']}: forecast would change by {r['change_pct']}% "
            f"({r[change_column]:+,.1f} Dth/day)"
        )
    return "\n".join(lines)


def _as_text(content):
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "").strip()


def _make_llm():
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY is not set in the terminal that started this app."
        )
    import httpx
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=BASE_URL,
        model=MODEL,
        api_key=api_key,
        http_client=httpx.Client(timeout=30, verify=VERIFY_SSL),
        max_retries=1,
        temperature=0.2,
    )


# ---------------------------------------------------------------- prompts
def build_prompt(report, state=None, tone=DEFAULT_TONE):
    location = f" for {state}" if state else ""
    length = TONES.get(tone, TONES[DEFAULT_TONE])
    return (
        "You are helping a gas demand analyst. Below are stress-test results for the "
        f"gas-day demand forecast{location} ({TARGET_GAS_DAY}). "
        f"Write a warning ({length}) "
        "in plain English: name the biggest risk, say which direction it pushes demand and "
        "what that could mean for gas purchasing (shortfall or surplus), and suggest one "
        "sensible action for the analyst to consider. Amounts are in Dth/day. "
        "Use ONLY the numbers given below. "
        "Do not invent figures. Note that scenarios mentioning 'assumed' are assumptions, "
        "not measured facts.\n\n"
        f"Results:\n{report_as_text(report)}"
    )


def build_briefing_prompt(reports, tone=DEFAULT_TONE):
    """reports: dict of state name -> report DataFrame."""
    length = TONES.get(tone, TONES[DEFAULT_TONE])
    blocks = "\n\n".join(
        f"{state}:\n{report_as_text(rep)}" for state, rep in reports.items()
    )
    return (
        "You are helping a gas demand analyst. Below are stress-test results for the "
        f"gas-day demand forecast ({TARGET_GAS_DAY}) in several states. "
        f"Write one combined briefing ({length}) in plain English: for each state name the "
        "biggest risk and its direction, then say which state needs attention first and "
        "suggest one sensible action. Amounts are in Dth/day. Use ONLY the numbers given "
        "below. Do not invent figures. Scenarios mentioning 'assumed' are assumptions, "
        "not measured facts.\n\n"
        f"Results:\n{blocks}"
    )


# ---------------------------------------------------------------- LLM calls
def call_llm(prompt):
    """Return the model's text, or None if no LLM is available.

    The key is read from LLM_API_KEY. If the call fails, the reason is
    stored in LAST_ERROR.
    """
    global LAST_ERROR
    LAST_ERROR = None
    try:
        llm = _make_llm()
        text = _as_text(llm.invoke(prompt).content)
        if text:
            return text
        LAST_ERROR = "The model returned an empty response."
        return None
    except Exception as exc:
        LAST_ERROR = str(exc)
        print(f"(LLM call failed: {exc}. Using the built-in template instead.)")
        return None


def ask_llm(question, context, history=()):
    """Answer an analyst question using only `context`. history: [(role, text)]."""
    global LAST_ERROR
    LAST_ERROR = None
    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        llm = _make_llm()
        messages = [
            SystemMessage(
                content=(
                    "You are an assistant for a gas demand analyst. Answer ONLY from the "
                    "data below. If the data does not contain the answer, say so. Do not "
                    "invent numbers. Amounts are in Dth/day. Scenarios mentioning "
                    "'assumed' are assumptions. Be concise (under 150 words).\n\n"
                    f"DATA:\n{context}"
                )
            )
        ]
        for role, content in list(history)[-6:]:
            messages.append(
                HumanMessage(content=content)
                if role == "user"
                else AIMessage(content=content)
            )
        messages.append(HumanMessage(content=question))
        text = _as_text(llm.invoke(messages).content)
        if text:
            return text
        LAST_ERROR = "The model returned an empty response."
        return None
    except Exception as exc:
        LAST_ERROR = str(exc)
        print(f"(LLM call failed: {exc})")
        return None


# ---------------------------------------------------------------- fact check
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUM_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)(\s?%)?")


def check_numbers(text, report, extra=()):
    """Return the numbers in `text` that do not appear in the report.

    `extra` is a list of other numbers that are allowed (e.g. the temperature
    shift or the nominations). Small whole numbers (10 or less) without a %
    sign are ignored. An empty list means every number was found in the data.
    """
    change_column = _change_column(report)
    allowed = []
    for col in ("forecast_dth", change_column, "change_pct"):
        if col in report:
            allowed.extend(abs(float(v)) for v in report[col])
    allowed.extend(abs(float(v)) for v in extra)

    unsupported = []
    for match in _NUM_RE.finditer(_DATE_RE.sub(" ", text)):
        raw, pct = match.group(1), match.group(2)
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        is_whole = "." not in raw
        if is_whole and value <= 10 and not pct:
            continue
        tolerance = 0.5 if is_whole else 0.051
        if not any(abs(value - a) <= tolerance for a in allowed):
            unsupported.append(raw.strip(",") + (pct.strip() if pct else ""))
    return unsupported


# ---------------------------------------------------------------- templates
def template_warning(report, state=None):
    """Rule-based fallback so the demo never breaks."""
    change_column = _change_column(report)
    top = report.iloc[0]
    up = report.loc[report["change_pct"].idxmax()]
    down = report.loc[report["change_pct"].idxmin()]
    location = f" for {state}" if state else ""
    return (
        f"Forecast risk{location} for gas day {TARGET_GAS_DAY}\n\n"
        f"Biggest risk: {top['scenario']} could move daily demand by "
        f"{top['change_pct']:+.1f}% ({top[change_column]:+,.1f} Dth/day).\n\n"
        f"Upside risk: under '{up['scenario']}', demand could be about "
        f"{up['change_pct']:.1f}% higher than forecast ({up[change_column]:+,.1f} Dth/day), "
        "which means a possible gas shortfall.\n"
        f"Downside risk: under '{down['scenario']}', demand could be about "
        f"{abs(down['change_pct']):.1f}% lower ({down[change_column]:+,.1f} Dth/day), "
        "which means a possible surplus.\n\n"
        f"Suggested action for the analyst to review: keep a reserve of roughly "
        f"{up[change_column]:,.1f} Dth/day of flexible supply and revisit the forecast if "
        "the weather outlook changes. Scenarios marked 'assumed' are assumptions, not measured facts."
    )


def template_briefing(reports):
    """Fallback combined briefing for several states."""
    lines = [f"Forecast risk briefing for gas day {TARGET_GAS_DAY}", ""]
    for state, rep in reports.items():
        col = _change_column(rep)
        top = rep.iloc[0]
        lines.append(
            f"{state}: biggest risk is {top['scenario']} "
            f"({top['change_pct']:+.1f}%, {top[col]:+,.1f} Dth/day)."
        )
    lines += [
        "",
        "Scenarios marked 'assumed' are assumptions, not measured facts.",
    ]
    return "\n".join(lines)


def main():
    report = pd.read_csv(REPORT_FILE)
    text = call_llm(build_prompt(report))
    source = "LLM"
    if text is None:
        text = template_warning(report)
        source = "built-in template"

    print(f"=== Analyst warning (written by: {source}) ===\n")
    print(text)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
