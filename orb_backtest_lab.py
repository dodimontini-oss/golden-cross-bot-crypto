"""
Opening Range Breakout (ORB) backtest - a genuinely different strategy
mechanism from Golden Cross (momentum/breakout off a fixed time window vs.
trend-following MA crossover). Rules as given by the user, from a TikTok
walkthrough (tradex_labs, #nq #futures #propfirmtrading):

  1. Mark the high/low of the first 5-minute candle of the regular session
     (9:30-9:35 AM America/New_York) - "the opening range."
  2. Wait for a later candle in the SAME session to CLOSE above the range
     high (go long) or below the range low (go short) - not an instant
     wick-touch breakout, a confirmed close beyond the range.
  3. Confluences to filter out chop, all requested explicitly:
       - ATR: used as the volatility reference for "normalize the range"
         below, rather than a separate standalone filter (the user listed
         both together and they're naturally the same mechanism: range
         width means nothing on its own without a volatility scale).
       - Relative volume: today's opening-range candle's volume vs. the
         trailing 20-session average volume of THAT SAME opening 5-minute
         slot (comparing today's open to typical opens, not to the whole
         day's average volume - the open is always the busiest 5 minutes
         of the day, so comparing to the day's average would pass almost
         every time and filter nothing).
       - Normalize the range: opening range width / daily ATR(14) - keeps
         the filter meaningful across QQQ's different volatility regimes
         (2020 vs 2026) instead of using a fixed point threshold.
       - Parameter optimization: swept explicitly below (relative-volume
         threshold x normalized-range band), against a no-filter BASELINE,
         same "test it, don't assume the confluence helps" approach this
         project uses for every other filter (GAPCONFIRM, CURCAP, ADX, ...).

Exit (per user's explicit choice - no session-close flatten): stop at the
OPPOSITE side of the opening range, target at RR_RATIO x that risk
distance. Position can carry past the session close if neither is hit yet.

INSTRUMENT: trades QQQ, not NQ futures directly - CME futures tick data
isn't available for free anywhere (checked: Databento/Barchart/CME are
paid; Yahoo Finance caps intraday data at 60 days). QQQ tracks the same
Nasdaq-100 index NQ is a derivative of, so the SHAPE of intraday price
action is highly correlated - this tests the strategy LOGIC, not real NQ
point-value P&L. Going live on actual NQ/MNQ futures would need a real
futures data+execution venue (Tradovate, IBKR, etc.), a separate build.

DATA: Alpaca's stock market data API (IEX feed, free tier - same account
already used for the crypto bot), 5-minute QQQ bars back to 2020-07-27
(~5 years) - the deepest free intraday history found (Yahoo: 60 days only).
Confirmed via check_alpaca_stock_data.py this account already has access.
Bars appear to be RTH-only (9:30-16:00 ET) based on the sample checked -
this means a carried position isn't marked-to-market bar-by-bar overnight
or pre-market; it's simply re-checked at the next available RTH bar. A
real gap risk exists live that this backtest can't see between sessions.

Run: python orb_backtest_lab.py

Environment variables required:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
"""

import logging
import os

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("orb-backtest-lab")

ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
DATA_URL = "https://data.alpaca.markets/v2/stocks/QQQ/bars"
HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

SYMBOL = "QQQ"
START = "2020-07-27T00:00:00Z"
END = "2026-08-27T00:00:00Z"

ATR_LEN = 14
RR_RATIO = 2.0
RISK_PER_TRADE_PCT = 1.0
STARTING_EQUITY = 10000.0
REL_VOL_LOOKBACK_DAYS = 20  # trailing sessions used to build "typical opening volume"

# Parameter sweep grid - "parameter optimization" per the user's rules.
REL_VOL_THRESHOLDS = [None, 1.0, 1.5, 2.0]          # None = no relative-volume filter
NORMALIZED_RANGE_BANDS = [None, (0.05, 0.40), (0.10, 0.60)]  # None = no range-normalization filter


def fetch_bars() -> pd.DataFrame:
    all_rows = []
    page_token = None
    while True:
        params = {"timeframe": "5Min", "start": START, "end": END, "limit": 10000, "feed": "iex"}
        if page_token:
            params["page_token"] = page_token
        resp = requests.get(DATA_URL, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        bars = data.get("bars", [])
        all_rows.extend(bars)
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


def build_daily_atr(df: pd.DataFrame) -> pd.Series:
    """Daily ATR(14) from the 5-min bars aggregated to daily OHLC - used as
    the volatility reference for normalizing the opening range's width."""
    daily = df.groupby("session_date").agg(open=("open", "first"), high=("high", "max"),
                                             low=("low", "min"), close=("close", "last"))
    prev_close = daily["close"].shift(1)
    tr = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - prev_close).abs(),
        (daily["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_LEN).mean()  # simple rolling mean is fine for a daily reference series
    return atr.shift(1)  # yesterday's ATR is what's known at today's open - no lookahead


def run_backtest(df: pd.DataFrame, daily_atr: pd.Series,
                  rel_vol_threshold, normalized_range_band) -> dict:
    equity = STARTING_EQUITY
    peak_equity = equity
    max_drawdown_pct = 0.0
    trades = []

    sessions = sorted(df["session_date"].unique())
    open_vol_by_session = {}  # session_date -> opening-range candle's volume, for the rel-vol lookback

    open_position = None  # dict: direction, entry, stop, target, units

    for i, session in enumerate(sessions):
        day_df = df[df["session_date"] == session].reset_index(drop=True)
        open_bar_rows = day_df[day_df["hm"] == "09:30"]
        if open_bar_rows.empty:
            continue
        open_bar = open_bar_rows.iloc[0]
        open_vol_by_session[session] = open_bar["volume"]

        # manage a position carried in from a prior session first - if it
        # closes today, that's it for today (no new entry the same session
        # off the same day's range - keeps this to one trade attempt per
        # session, and avoids re-scanning bars already consumed managing
        # the carried position for a fresh breakout).
        if open_position is not None:
            for _, row in day_df.iterrows():
                d = open_position
                if d["direction"] == "LONG":
                    hit_stop = row["low"] <= d["stop"]
                    hit_target = row["high"] >= d["target"]
                else:
                    hit_stop = row["high"] >= d["stop"]
                    hit_target = row["low"] <= d["target"]
                if hit_stop or hit_target:
                    exit_price = d["stop"] if hit_stop else d["target"]
                    pnl = (exit_price - d["entry"]) * d["units"] if d["direction"] == "LONG" \
                        else (d["entry"] - exit_price) * d["units"]
                    equity += pnl
                    peak_equity = max(peak_equity, equity)
                    dd = ((peak_equity - equity) / peak_equity * 100) if peak_equity > 0 else 0
                    max_drawdown_pct = max(max_drawdown_pct, dd)
                    trades.append({"pnl": pnl, "session": session})
                    open_position = None
                    break
            continue

        atr = daily_atr.get(session)
        if pd.isna(atr) or atr is None or atr <= 0:
            continue

        range_high, range_low = open_bar["high"], open_bar["low"]
        range_width = range_high - range_low
        if range_width <= 0:
            continue

        if normalized_range_band is not None:
            normalized = range_width / atr
            lo, hi = normalized_range_band
            if not (lo <= normalized <= hi):
                continue

        if rel_vol_threshold is not None:
            past_vols = [open_vol_by_session[s] for s in sessions[max(0, i - REL_VOL_LOOKBACK_DAYS):i]
                         if s in open_vol_by_session]
            if len(past_vols) < 5:
                continue  # not enough history yet to judge "relative" volume
            avg_open_vol = sum(past_vols) / len(past_vols)
            if avg_open_vol <= 0 or (open_bar["volume"] / avg_open_vol) < rel_vol_threshold:
                continue

        # watch the rest of THIS session for a confirmed close beyond the range
        rest = day_df[day_df["hm"] > "09:30"]
        for _, row in rest.iterrows():
            direction = None
            if row["close"] > range_high:
                direction = "LONG"
            elif row["close"] < range_low:
                direction = "SHORT"
            if direction is None:
                continue

            entry = row["close"]
            stop = range_low if direction == "LONG" else range_high
            stop_distance = abs(entry - stop)
            if stop_distance <= 0:
                break
            target = entry + stop_distance * RR_RATIO if direction == "LONG" \
                else entry - stop_distance * RR_RATIO
            risk_amount = equity * RISK_PER_TRADE_PCT / 100
            units = risk_amount / stop_distance
            open_position = {"direction": direction, "entry": entry, "stop": stop,
                              "target": target, "units": units}
            break  # only one entry attempt per session

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    win_rate = (len(wins) / len(trades) * 100) if trades else 0
    net_pnl_pct = (equity - STARTING_EQUITY) / STARTING_EQUITY * 100

    return {
        "trades": len(trades), "win_rate": win_rate, "profit_factor": profit_factor,
        "net_pnl_pct": net_pnl_pct, "max_drawdown_pct": max_drawdown_pct, "final_equity": equity,
    }


def main():
    log.info("Fetching QQQ 5-minute bars (%s to %s) from Alpaca (IEX feed)...", START, END)
    df = fetch_bars()
    log.info("Got %d 5-minute bars across %d sessions.", len(df), df["session_date"].nunique())

    daily_atr = build_daily_atr(df)

    results = []
    for rel_vol in REL_VOL_THRESHOLDS:
        for band in NORMALIZED_RANGE_BANDS:
            label = f"relvol={'off' if rel_vol is None else f'{rel_vol:.1f}x'}, " \
                    f"range/ATR={'off' if band is None else f'{band[0]:.2f}-{band[1]:.2f}'}"
            log.info("Running: %s ...", label)
            r = run_backtest(df, daily_atr, rel_vol, band)
            r["label"] = label
            r["is_baseline"] = rel_vol is None and band is None
            results.append(r)

    print("\n" + "=" * 105)
    print(f"{'Config':<42} {'Trades':>7} {'Win%':>7} {'ProfitFactor':>13} {'Net P&L%':>10} {'MaxDD%':>8}")
    print("=" * 105)
    for r in results:
        marker = "  <-- BASELINE (no confluences)" if r["is_baseline"] else ""
        print(f"{r['label']:<42} {r['trades']:>7} {r['win_rate']:>6.1f}% "
              f"{r['profit_factor']:>13.3f} {r['net_pnl_pct']:>+9.1f}% {r['max_drawdown_pct']:>7.1f}%{marker}")
    print("=" * 105)
    print(f"\nStarting equity: ${STARTING_EQUITY:,.0f}  |  QQQ 5-min bars, {df['session_date'].nunique()} sessions "
          f"({df['session_date'].min()} to {df['session_date'].max()}).")
    print("Proxy for NQ futures (see module docstring) - tests strategy LOGIC, not real futures P&L.")
    print("Exit: stop = opposite side of opening range, target = 2:1 R:R. No session-close flatten "
          "(positions can carry past the close if neither stop nor target is hit yet).\n")


if __name__ == "__main__":
    main()
