"""
Same exact strategy/engine as orb_multi_timeframe_lab.py, run on SPY instead
of QQQ - user asked whether this ORB approach generalizes to S&P 500 (ES/MES
futures) or is a Nasdaq-100-specific edge. Rather than assume either way,
this re-runs the identical backtest (same rules, same fee model, same
walk-forward split, same 4 ORB windows x 2 configs) on SPY, the S&P 500's
equity-ETF proxy (same Alpaca IEX data source already confirmed to have
deep history - see check_alpaca_stock_data.py).

See orb_multi_timeframe_lab.py's docstring for the full strategy spec and
caveats (QQQ/SPY-as-futures-proxy, RTH-only bars, etc.) - identical here,
just a different underlying symbol.

Run: python orb_spy_multi_timeframe_lab.py

Environment variables required:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
"""

import logging
import os

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("orb-spy-multi-timeframe-lab")

ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
DATA_URL = "https://data.alpaca.markets/v2/stocks/SPY/bars"
HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

START = "2020-07-27T00:00:00Z"
END = "2026-08-27T00:00:00Z"

ATR_LEN = 14
RR_RATIO = 2.0
RISK_PER_TRADE_PCT = 1.0
STARTING_EQUITY = 10000.0
REL_VOL_LOOKBACK_DAYS = 20
FEE_PCT_PER_SIDE = 0.0001

ORB_WINDOWS_MINUTES = [5, 15, 30, 60]
CONFIGS = [
    ("BASELINE", None),
    ("relvol>=1.5x", 1.5),
]


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
    daily = df.groupby("session_date").agg(open=("open", "first"), high=("high", "max"),
                                             low=("low", "min"), close=("close", "last"))
    prev_close = daily["close"].shift(1)
    tr = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - prev_close).abs(),
        (daily["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_LEN).mean()
    return atr.shift(1)


def _orb_end_hm(window_minutes: int) -> str:
    n_bars = window_minutes // 5
    end_minute_offset = (n_bars - 1) * 5
    hour = 9 + (30 + end_minute_offset) // 60
    minute = (30 + end_minute_offset) % 60
    return f"{hour:02d}:{minute:02d}"


def run_backtest(df: pd.DataFrame, daily_atr: pd.Series, orb_window_minutes: int,
                  rel_vol_threshold, fee_pct_per_side: float,
                  sessions_filter=None) -> dict:
    equity = STARTING_EQUITY
    peak_equity = equity
    max_drawdown_pct = 0.0
    trades = []

    orb_end_hm = _orb_end_hm(orb_window_minutes)
    sessions = sorted(df["session_date"].unique())
    if sessions_filter is not None:
        sessions = [s for s in sessions if s in sessions_filter]
    open_vol_by_session = {}

    open_position = None

    for i, session in enumerate(sessions):
        day_df = df[df["session_date"] == session].reset_index(drop=True)
        orb_bars = day_df[(day_df["hm"] >= "09:30") & (day_df["hm"] <= orb_end_hm)]
        if orb_bars.empty:
            continue
        range_high, range_low = orb_bars["high"].max(), orb_bars["low"].min()
        orb_volume = orb_bars["volume"].sum()
        open_vol_by_session[session] = orb_volume

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
                    gross_pnl = (exit_price - d["entry"]) * d["units"] if d["direction"] == "LONG" \
                        else (d["entry"] - exit_price) * d["units"]
                    exit_fee = exit_price * d["units"] * fee_pct_per_side
                    pnl = gross_pnl - exit_fee
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

        range_width = range_high - range_low
        if range_width <= 0:
            continue

        if rel_vol_threshold is not None:
            past_vols = [open_vol_by_session[s] for s in sessions[max(0, i - REL_VOL_LOOKBACK_DAYS):i]
                         if s in open_vol_by_session]
            if len(past_vols) < 5:
                continue
            avg_open_vol = sum(past_vols) / len(past_vols)
            if avg_open_vol <= 0 or (orb_volume / avg_open_vol) < rel_vol_threshold:
                continue

        rest = day_df[day_df["hm"] > orb_end_hm]
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
            entry_fee = entry * units * fee_pct_per_side
            equity -= entry_fee
            open_position = {"direction": direction, "entry": entry, "stop": stop,
                              "target": target, "units": units}
            break

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
    log.info("Fetching SPY 5-minute bars (%s to %s) from Alpaca (IEX feed)...", START, END)
    df = fetch_bars()
    sessions_all = sorted(df["session_date"].unique())
    log.info("Got %d 5-minute bars across %d sessions.", len(df), len(sessions_all))

    daily_atr = build_daily_atr(df)

    results = []
    for window in ORB_WINDOWS_MINUTES:
        for label, rel_vol in CONFIGS:
            full_label = f"{window}min ORB, {label}"
            log.info("Running: %s ...", full_label)
            r = run_backtest(df, daily_atr, window, rel_vol, FEE_PCT_PER_SIDE)
            r["label"] = full_label
            results.append(r)

    print("\n" + "=" * 100)
    print("SPY (S&P 500 proxy) - MULTI-TIMEFRAME SWEEP (with realistic ~0.01%/side commission)")
    print("=" * 100)
    print(f"{'Config':<28} {'Trades':>7} {'Win%':>7} {'ProfitFactor':>13} {'Net P&L%':>10} {'MaxDD%':>8}")
    print("-" * 100)
    for r in results:
        print(f"{r['label']:<28} {r['trades']:>7} {r['win_rate']:>6.1f}% "
              f"{r['profit_factor']:>13.3f} {r['net_pnl_pct']:>+9.1f}% {r['max_drawdown_pct']:>7.1f}%")
    print("=" * 100)

    split_idx = len(sessions_all) // 2
    early_sessions = set(sessions_all[:split_idx])
    late_sessions = set(sessions_all[split_idx:])
    split_date = sessions_all[split_idx]

    print(f"\nSPY WALK-FORWARD CHECK, 5-minute ORB (early: {sessions_all[0]} to {sessions_all[split_idx-1]}"
          f"  |  late: {split_date} to {sessions_all[-1]}), with fees")
    print("=" * 100)
    print(f"{'Config':<28} {'Trades':>7} {'Win%':>7} {'ProfitFactor':>13} {'Net P&L%':>10} {'MaxDD%':>8}")
    print("-" * 100)
    for label, rel_vol in CONFIGS:
        for period_name, sess_set in [("EARLY", early_sessions), ("LATE", late_sessions)]:
            r = run_backtest(df, daily_atr, 5, rel_vol, FEE_PCT_PER_SIDE, sessions_filter=sess_set)
            print(f"{label + ' - ' + period_name:<28} {r['trades']:>7} {r['win_rate']:>6.1f}% "
                  f"{r['profit_factor']:>13.3f} {r['net_pnl_pct']:>+9.1f}% {r['max_drawdown_pct']:>7.1f}%")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
