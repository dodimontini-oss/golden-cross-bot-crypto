"""
ORB + Fair-Value-Gap retest variant, requested after the user watched more
ORB videos: instead of entering the moment the opening range breaks (the
current live bots' rule), wait for the breakout to leave a 3-candle Fair
Value Gap (FVG) in the breakout's direction, then wait for price to pull
back and retest that gap before entering - the idea being a better average
entry price / tighter effective risk than chasing the initial breakout close.

FVG definition (standard 3-candle ICT/SMC definition, bars A, B, C in time
order - B is the displacement candle, only A and C's wicks matter for the
gap itself):
    Bullish FVG:  A.high < C.low   -> zone = [A.high, C.low]
    Bearish FVG:  A.low  > C.high  -> zone = [C.high, A.low]

The user wanted three mechanics questions answered by testing BOTH options
rather than picking one, so this runs the full 2x2x2 = 8-combo grid:

  1. Which candles define the FVG to watch for -
     - "breakout_only": check exactly once, using the breakout's own 3-bar
       group (bar before the confirmed-close bar, the confirmed-close bar,
       bar after). No trade that day if this specific triplet doesn't gap.
     - "scan_forward": keep checking every rolling 3-bar window starting
       from the breakout bar, moving forward through the rest of the
       session, taking the FIRST one that qualifies. This is a strict
       superset of "breakout_only" (same first check, then keeps looking).

  2. What triggers entry once there's a zone to watch -
     - "first_touch": enter the instant any later bar's wick trades back
       into the zone, filled at the zone's NEAR edge (a resting limit order
       at the edge closest to where price is retracing from - the most
       likely fill price, ignoring further slippage into the zone).
     - "confirmation_candle": require a bar that both touches the zone AND
       closes back in the breakout's direction before entering, filled at
       that bar's close. Fewer/later entries, no guarantee it ever arrives.

  3. Stop/target basis -
     - "orb_stop": stop stays the opposite side of the opening range (same
       rule the live bots use today), target = 2:1 R:R from the NEW
       (retest) entry price.
     - "fvg_edge_stop": stop sits just beyond the FAR edge of the FVG zone
       itself (the tight ICT-standard stop), target = 2:1 R:R from there.

A stop-sanity guard (added to the live bots 2026-08-28 after a real stale-
signal incident - see project_orb_futures_strategy_findings memory) is
applied here too: an entry is discarded if the computed stop is already on
the wrong side of the entry price (would be an instant stop-out) - keeps
the backtest from crediting trades a real broker would reject.

A trade only ever gets ONE session to develop (breakout -> FVG -> retest
all within the same day's remaining bars) - if the retest never comes by
session close, no trade that day. This is a simplifying assumption (matches
the existing "one attempt per session" ORB philosophy) - carrying a pending
FVG watch across a session boundary is a possible follow-up if results here
look promising, not built here.

Same instrument/data source as the other ORB labs - QQQ 5-minute bars via
Alpaca's IEX feed, 2020-07-27 onward (see orb_backtest_lab.py's docstring
for the full QQQ-as-NQ-proxy rationale, not repeated here). Tests the
5-minute opening-range window only (matches orb-bot-5min, the primary live
deployment) - the 15-minute variant is a natural follow-up if this concept
proves out, not run here to keep the grid size sane.

Run: python orb_fvg_retest_lab.py

Environment variables required:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
"""

import logging
import os

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("orb-fvg-retest-lab")

ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
DATA_URL = "https://data.alpaca.markets/v2/stocks/QQQ/bars"
HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

START = "2020-07-27T00:00:00Z"
END = "2026-08-29T00:00:00Z"

ORB_WINDOW_MINUTES = 5  # matches orb-bot-5min, the primary live deployment
RR_RATIO = 2.0
RISK_PER_TRADE_PCT = 1.0
STARTING_EQUITY = 10000.0
FEE_PCT_PER_SIDE = 0.0001  # 0.01%/side - same conservative estimate used in the other ORB labs

# Already-established reference point (see project_orb_futures_strategy_findings
# memory) - immediate entry on breakout close, no FVG, with the same fee model.
BASELINE_REFERENCE = {
    "label": "5min ORB, immediate entry (no FVG) - REFERENCE", "trades": 730,
    "win_rate": 43.2, "profit_factor": 1.353, "net_pnl_pct": 415.5, "max_drawdown_pct": 13.3,
}

FVG_WINDOW_MODES = ["breakout_only", "scan_forward"]
ENTRY_TRIGGER_MODES = ["first_touch", "confirmation_candle"]
STOP_MODES = ["orb_stop", "fvg_edge_stop"]


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


def _orb_end_hm(window_minutes: int) -> str:
    n_bars = window_minutes // 5
    end_minute_offset = (n_bars - 1) * 5
    hour = 9 + (30 + end_minute_offset) // 60
    minute = (30 + end_minute_offset) % 60
    return f"{hour:02d}:{minute:02d}"


def _check_fvg(bar_a, bar_c, direction):
    """bar_a/bar_c are the outer two bars of a 3-bar window - only their
    wicks matter for the gap, the middle (displacement) bar isn't checked."""
    if direction == "LONG" and bar_a["high"] < bar_c["low"]:
        return bar_a["high"], bar_c["low"]
    if direction == "SHORT" and bar_a["low"] > bar_c["high"]:
        return bar_c["high"], bar_a["low"]
    return None


def find_fvg(day_df: pd.DataFrame, breakout_idx: int, direction: str, mode: str):
    """Returns (zone_low, zone_high, end_idx) for the first qualifying FVG at
    or after the breakout, or None if none forms before the session ends."""
    n = len(day_df)
    if mode == "breakout_only":
        a_idx, c_idx = breakout_idx - 1, breakout_idx + 1
        if a_idx < 0 or c_idx >= n:
            return None
        zone = _check_fvg(day_df.iloc[a_idx], day_df.iloc[c_idx], direction)
        return (zone[0], zone[1], c_idx) if zone else None
    for a_idx in range(max(0, breakout_idx - 1), n - 2):
        c_idx = a_idx + 2
        zone = _check_fvg(day_df.iloc[a_idx], day_df.iloc[c_idx], direction)
        if zone:
            return zone[0], zone[1], c_idx
    return None


def find_retest_entry(day_df: pd.DataFrame, zone_low: float, zone_high: float, fvg_end_idx: int,
                       direction: str, mode: str):
    """Watches bars strictly after the FVG forms for a retest. Returns
    (entry_price, entry_idx) or None if never retested this session."""
    entry_limit = zone_high if direction == "LONG" else zone_low  # near edge, where price arrives from
    for idx in range(fvg_end_idx + 1, len(day_df)):
        row = day_df.iloc[idx]
        touched = row["low"] <= zone_high and row["high"] >= zone_low
        if not touched:
            continue
        if mode == "first_touch":
            return entry_limit, idx
        bullish, bearish = row["close"] > row["open"], row["close"] < row["open"]
        if (direction == "LONG" and bullish) or (direction == "SHORT" and bearish):
            return row["close"], idx
        # touched but didn't confirm - keep watching later bars
    return None


def compute_stop_target(direction: str, entry: float, range_high: float, range_low: float,
                         zone_low: float, zone_high: float, stop_mode: str):
    stop = (range_low if stop_mode == "orb_stop" else zone_low) if direction == "LONG" \
        else (range_high if stop_mode == "orb_stop" else zone_high)
    # Stop-sanity guard (see module docstring) - discard if already invalid.
    if direction == "LONG" and entry <= stop:
        return None
    if direction == "SHORT" and entry >= stop:
        return None
    stop_distance = abs(entry - stop)
    target = entry + stop_distance * RR_RATIO if direction == "LONG" else entry - stop_distance * RR_RATIO
    return stop, target, stop_distance


def _close_position(open_position, row, fee_pct_per_side, equity, peak_equity, max_dd, trades, session):
    d = open_position
    if d["direction"] == "LONG":
        hit_stop, hit_target = row["low"] <= d["stop"], row["high"] >= d["target"]
    else:
        hit_stop, hit_target = row["high"] >= d["stop"], row["low"] <= d["target"]
    if not (hit_stop or hit_target):
        return None, equity, peak_equity, max_dd
    exit_price = d["stop"] if hit_stop else d["target"]
    gross_pnl = (exit_price - d["entry"]) * d["units"] if d["direction"] == "LONG" \
        else (d["entry"] - exit_price) * d["units"]
    pnl = gross_pnl - exit_price * d["units"] * fee_pct_per_side
    equity += pnl
    peak_equity = max(peak_equity, equity)
    max_dd = max(max_dd, ((peak_equity - equity) / peak_equity * 100) if peak_equity > 0 else 0)
    trades.append({"pnl": pnl, "session": session})
    return "closed", equity, peak_equity, max_dd


def run_backtest(df: pd.DataFrame, fvg_mode: str, entry_mode: str, stop_mode: str,
                  fee_pct_per_side: float, sessions_filter=None) -> dict:
    equity = peak_equity = STARTING_EQUITY
    max_dd = 0.0
    trades = []
    breakout_days = fvg_days = retest_days = 0

    orb_end_hm = _orb_end_hm(ORB_WINDOW_MINUTES)
    sessions = sorted(df["session_date"].unique())
    if sessions_filter is not None:
        sessions = [s for s in sessions if s in sessions_filter]

    open_position = None

    for session in sessions:
        day_df = df[df["session_date"] == session].reset_index(drop=True)

        if open_position is not None:
            for _, row in day_df.iterrows():
                status, equity, peak_equity, max_dd = _close_position(
                    open_position, row, fee_pct_per_side, equity, peak_equity, max_dd, trades, session)
                if status == "closed":
                    open_position = None
                    break
            continue

        orb_bars = day_df[(day_df["hm"] >= "09:30") & (day_df["hm"] <= orb_end_hm)]
        if orb_bars.empty:
            continue
        range_high, range_low = orb_bars["high"].max(), orb_bars["low"].min()
        if range_high - range_low <= 0:
            continue

        rest = day_df[day_df["hm"] > orb_end_hm]
        if rest.empty:
            continue
        direction = breakout_idx = None
        for idx, row in rest.iterrows():
            if row["close"] > range_high:
                direction, breakout_idx = "LONG", idx
                break
            elif row["close"] < range_low:
                direction, breakout_idx = "SHORT", idx
                break
        if direction is None:
            continue
        breakout_days += 1

        fvg = find_fvg(day_df, breakout_idx, direction, fvg_mode)
        if fvg is None:
            continue
        zone_low, zone_high, fvg_end_idx = fvg
        fvg_days += 1

        retest = find_retest_entry(day_df, zone_low, zone_high, fvg_end_idx, direction, entry_mode)
        if retest is None:
            continue
        entry, entry_idx = retest
        retest_days += 1

        st = compute_stop_target(direction, entry, range_high, range_low, zone_low, zone_high, stop_mode)
        if st is None:
            continue
        stop, target, stop_distance = st

        risk_amount = equity * RISK_PER_TRADE_PCT / 100
        units = risk_amount / stop_distance
        equity -= entry * units * fee_pct_per_side
        open_position = {"direction": direction, "entry": entry, "stop": stop, "target": target, "units": units}

        for _, row in day_df.iloc[entry_idx + 1:].iterrows():
            status, equity, peak_equity, max_dd = _close_position(
                open_position, row, fee_pct_per_side, equity, peak_equity, max_dd, trades, session)
            if status == "closed":
                open_position = None
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
        "net_pnl_pct": net_pnl_pct, "max_drawdown_pct": max_dd, "final_equity": equity,
        "breakout_days": breakout_days, "fvg_days": fvg_days, "retest_days": retest_days,
    }


def print_row(label, r):
    small = " (small sample - treat cautiously)" if 0 < r["trades"] < 30 else ""
    print(f"{label:<52} {r['trades']:>7} {r['win_rate']:>6.1f}% "
          f"{r['profit_factor']:>13.3f} {r['net_pnl_pct']:>+9.1f}% {r['max_drawdown_pct']:>7.1f}%{small}")


def main():
    log.info("Fetching QQQ 5-minute bars (%s to %s) from Alpaca (IEX feed)...", START, END)
    df = fetch_bars()
    sessions_all = sorted(df["session_date"].unique())
    log.info("Got %d 5-minute bars across %d sessions.", len(df), len(sessions_all))

    combos = [(f, e, s) for f in FVG_WINDOW_MODES for e in ENTRY_TRIGGER_MODES for s in STOP_MODES]

    print("\n" + "=" * 110)
    print("PART 1: FULL-SAMPLE GRID, 5-min ORB + FVG retest (8 combos), with realistic ~0.01%/side commission")
    print("=" * 110)
    print(f"{'Config':<52} {'Trades':>7} {'Win%':>7} {'ProfitFactor':>13} {'Net P&L%':>10} {'MaxDD%':>8}")
    print("-" * 110)
    print_row(BASELINE_REFERENCE["label"], BASELINE_REFERENCE)
    print("-" * 110)
    full_results = {}
    for fvg_mode, entry_mode, stop_mode in combos:
        label = f"FVG={fvg_mode} / entry={entry_mode} / stop={stop_mode}"
        log.info("Running: %s ...", label)
        r = run_backtest(df, fvg_mode, entry_mode, stop_mode, FEE_PCT_PER_SIDE)
        full_results[(fvg_mode, entry_mode, stop_mode)] = r
        print_row(label, r)
    print("=" * 110)

    print("\nFunnel (how selective each combo is - breakout days -> FVG formed -> retest happened -> traded):")
    print("-" * 110)
    for (fvg_mode, entry_mode, stop_mode), r in full_results.items():
        label = f"FVG={fvg_mode} / entry={entry_mode} / stop={stop_mode}"
        print(f"{label:<52} breakouts={r['breakout_days']:>4}  fvg_formed={r['fvg_days']:>4}  "
              f"retested={r['retest_days']:>4}  traded={r['trades']:>4}")
    print("-" * 110)

    # ---- walk-forward split on all 8 combos ----
    split_idx = len(sessions_all) // 2
    early_sessions = set(sessions_all[:split_idx])
    late_sessions = set(sessions_all[split_idx:])
    split_date = sessions_all[split_idx]

    print(f"\nPART 2: WALK-FORWARD CHECK (early: {sessions_all[0]} to {sessions_all[split_idx-1]}"
          f"  |  late: {split_date} to {sessions_all[-1]}), with fees")
    print("=" * 110)
    print(f"{'Config':<52} {'Trades':>7} {'Win%':>7} {'ProfitFactor':>13} {'Net P&L%':>10} {'MaxDD%':>8}")
    print("-" * 110)
    for fvg_mode, entry_mode, stop_mode in combos:
        base_label = f"FVG={fvg_mode} / entry={entry_mode} / stop={stop_mode}"
        for period_name, sess_set in [("EARLY", early_sessions), ("LATE", late_sessions)]:
            r = run_backtest(df, fvg_mode, entry_mode, stop_mode, FEE_PCT_PER_SIDE, sessions_filter=sess_set)
            print_row(f"{base_label} - {period_name}", r)
    print("=" * 110)
    print("\nA combo only deserves trust if PF > 1.0 in BOTH halves AND the full-sample trade count is large")
    print("enough to not be noise (see print_row's small-sample flag) - pooling/cross-validation rules from")
    print("project methodology memory apply here same as anywhere else.\n")


if __name__ == "__main__":
    main()
