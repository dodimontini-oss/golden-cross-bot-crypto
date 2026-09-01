"""
Targeted follow-up to a live trade-history check (2026-09-01): both live ORB
bots (5-min and 15-min) lost their only two closed trades so far - a SHORT
breakout on 2026-08-31 and a LONG breakout on 2026-09-01, both stopped out.
The live bots run BASELINE (no confluence filter) - the user's explicit
choice over the backtest-recommended relvol>=1.5x filter (see
project_orb_futures_strategy_findings memory), which was built specifically
to skip low-conviction breakouts.

This checks a concrete, falsifiable question with real data rather than
speculating: were these two specific losing breakouts on LOW relative
volume - i.e., exactly the kind of trade relvol>=1.5x would have skipped?
Uses the same relvol definition as the original backtest (today's
opening-range volume vs. the trailing 20-session average volume in that
same opening slot).

Run via workflow_dispatch in golden-cross-bot-crypto (already has
Alpaca stock-data-capable credentials - see reference_github_repos memory).

Environment variables required:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
"""

import os

import pandas as pd
import requests

ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
DATA_BASE_URL = "https://data.alpaca.markets"
HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

SYMBOL = "QQQ"
REL_VOL_LOOKBACK_DAYS = 20
REL_VOL_THRESHOLD = 1.5

# The two actual live losing trades to check, both window lengths.
CHECKS = [
    ("2026-08-31", 5, "SHORT"),
    ("2026-08-31", 15, "SHORT"),
    ("2026-09-01", 5, "LONG"),
    ("2026-09-01", 15, "LONG"),
]


def get_bars(start, end):
    all_rows = []
    page_token = None
    while True:
        params = {"timeframe": "5Min", "start": start, "end": end, "limit": 10000, "feed": "iex"}
        if page_token:
            params["page_token"] = page_token
        resp = requests.get(f"{DATA_BASE_URL}/v2/stocks/{SYMBOL}/bars", headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        all_rows.extend(data.get("bars", []))
        page_token = data.get("next_page_token")
        if not page_token:
            break
    df = pd.DataFrame(all_rows)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df["time_et"] = df["t"].dt.tz_convert("America/New_York")
    df["session_date"] = df["time_et"].dt.date
    df["hm"] = df["time_et"].dt.strftime("%H:%M")
    return df.sort_values("t").reset_index(drop=True)


def orb_end_hm(window_minutes):
    n_bars = window_minutes // 5
    end_minute_offset = (n_bars - 1) * 5
    hour = 9 + (30 + end_minute_offset) // 60
    minute = (30 + end_minute_offset) % 60
    return f"{hour:02d}:{minute:02d}"


def main():
    # ~45 trading days back covers 20-session lookback plus slack for holidays/weekends
    df = get_bars("2026-06-20T00:00:00Z", "2026-09-02T00:00:00Z")
    sessions = sorted(df["session_date"].unique())
    print(f"Loaded {len(df)} bars across {len(sessions)} sessions ({sessions[0]} to {sessions[-1]})\n")
    print("=" * 100)

    for date_str, window, direction in CHECKS:
        target_date = pd.Timestamp(date_str).date()
        if target_date not in sessions:
            print(f"{date_str} ({window}min, {direction}): session not found in fetched range - skipping")
            continue
        idx = sessions.index(target_date)
        end_hm = orb_end_hm(window)

        # today's opening-range volume
        day_df = df[df["session_date"] == target_date]
        orb_bars = day_df[(day_df["hm"] >= "09:30") & (day_df["hm"] <= end_hm)]
        today_vol = orb_bars["volume"].sum()

        # trailing 20 sessions' volume in that same opening slot
        past_dates = sessions[max(0, idx - REL_VOL_LOOKBACK_DAYS):idx]
        past_vols = []
        for d in past_dates:
            pdf = df[df["session_date"] == d]
            pbars = pdf[(pdf["hm"] >= "09:30") & (pdf["hm"] <= end_hm)]
            if not pbars.empty:
                past_vols.append(pbars["volume"].sum())

        if len(past_vols) < 5:
            print(f"{date_str} ({window}min, {direction}): only {len(past_vols)} prior sessions available - "
                  f"too few for a reliable average, skipping")
            continue

        avg_vol = sum(past_vols) / len(past_vols)
        relvol = today_vol / avg_vol if avg_vol > 0 else float("nan")
        would_pass = relvol >= REL_VOL_THRESHOLD
        verdict = "WOULD HAVE TRADED (relvol filter passes)" if would_pass else \
                  "WOULD HAVE BEEN SKIPPED (relvol filter blocks it)"
        print(f"{date_str}  {window:>2}min ORB  {direction:<5}  today_vol={today_vol:>10.0f}  "
              f"20d_avg_vol={avg_vol:>10.0f}  relvol={relvol:.3f}x  (threshold {REL_VOL_THRESHOLD}x)")
        print(f"    -> {verdict}")

    print("=" * 100)
    print("\nIf both losing trades show relvol < 1.5x, that's real evidence the declined confluence filter")
    print("would have avoided both live losses - not proof it's a better long-run choice (n=2), but a concrete")
    print("data point for the user to weigh against the higher BASELINE total-return the backtest showed.\n")


if __name__ == "__main__":
    main()
