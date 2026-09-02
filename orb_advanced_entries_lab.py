"""
Three new 15-minute ORB entry mechanics, requested after the FVG-retest
variant (see project_orb_futures_strategy_findings memory, "2026-08-29:
FVG-retest variant tested and rejected") was rejected for using the WRONG
risk model - that test derived the stop from a fixed level (ORB range or
FVG edge) and let the target float at 2:1 R:R from there. This test inverts
that: the TARGET is fixed first (a real technical level), and the STOP is
back-solved so that reaching the target is worth exactly RR_RATIO (2:1).
That is a materially different mechanic from the rejected test, which is
exactly the bar the memory note set for re-testing this territory.

All three bots share: 15-minute opening range (09:30-09:45 ET), a confirmed
CLOSE beyond the range (not a wick) to establish breakout direction, and a
single attempt per session (if the entry condition never triggers by session
close, no trade that day - matches the existing ORB bots' convention).
Positions may carry across a session boundary if neither stop nor target has
been hit by the close (matches the live bots - "no session-close flatten").

BOT 1 - "Retest": wait for price to pull back and retest the broken range
boundary (range_high for a LONG, range_low for a SHORT) before entering,
instead of chasing the breakout's own close. Entry fills at the level itself.

BOT 2 - "FVG/OB retracement": wait for price to retrace back INTO a Fair
Value Gap or Order Block that formed INSIDE the opening range itself (using
only the 3 five-minute bars that make up the 15-min OR window) - a
DIFFERENT zone than the rejected test, which looked for FVGs formed by the
breakout's own displacement candles AFTER the range. FVG is checked first
(bar0 vs bar2 of the OR window gapping in the breakout's direction); if none,
falls back to an Order Block (the last opposite-colored OR candle, its full
[low, high] as the zone). Entry fills at the zone's near edge (first-touch
convention, same as the earlier FVG lab).

BOT 3 - "Full retracement to opposite zone": wait for price to retrace ALL
THE WAY back through the entire opening range to the zone at the OPPOSITE
extreme (the single OR bar that set range_low for a LONG / range_high for a
SHORT - a demand/supply zone at the far side of the range). Two geometry
variants tested per your request:
  - "zone_a" (literal reading of your spec): entry at the zone's FAR edge
    (the deepest point of the retracement), target the zone's NEAR edge - a
    quick bounce off the zone itself.
  - "zone_b" (alternate reading): entry at the zone's NEAR edge (first
    touch), target back at the range's own breakout boundary - a
    continuation trade, using more room than the zone's own width.

TP KEY-LEVEL - tested all three ways per your request, for bots 1 and 2 only
(bot 3's target is the zone itself, not a key level):
  - "pdh_pdl": previous REGULAR SESSION's high (LONG) / low (SHORT).
  - "premarket": TODAY's pre-market (bars before 09:30 ET) high/low. NOTE:
    diagnostic output at the top of the run reports how many bars per
    session fall before 09:30 - if Alpaca's IEX feed only ever returns
    regular-session bars (unconfirmed until this actually runs), this
    variant will show ~0 trades across the board, which is a data-coverage
    finding, not a strategy finding - read the diagnostic line before
    trusting (or dismissing) this variant's numbers.
  - "measured_move": range_high + 2x the opening range's own width (LONG) /
    range_low - 2x the width (SHORT) - a projection with no reference to
    prior sessions at all.

Risk model (all 8 configs): TP is set first per the rules above, then
stop_distance = abs(entry - TP) / RR_RATIO, SL = entry -/+ stop_distance.
A trade is DISCARDED (not taken) if: the key level ends up on the wrong side
of entry (no real reward available), or stop_distance falls below a safety
floor of 5% of the session's daily ATR(14). That floor is NOT in your
original spec - it's added because the rejected FVG test found that a
razor-thin zone-derived stop can blow up position size under fixed-fraction
risk sizing (92-100% drawdown in 5 of 8 combos there). Since bot 3's target
here is a raw zone width (same failure mode risk as that rejected test), and
bots 1/2's stop is *derived from* the zone/key-level distance, the floor is
cheap insurance against the same trap re-appearing quietly. If it binds
often, it'll show up in each config's funnel print as a real filtered count,
not silently.

Same instrument/data as the other ORB labs - QQQ 5-minute bars via Alpaca's
IEX feed, 2020-07-27 onward (see orb_backtest_lab.py for the QQQ-as-NQ-proxy
rationale). 15-minute window only (matches orb-bot-15min, and is the only
window you specified for these three).

Run: python orb_advanced_entries_lab.py

Environment variables required:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
"""

import logging
import os

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("orb-advanced-entries-lab")

ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
DATA_URL = "https://data.alpaca.markets/v2/stocks/QQQ/bars"
HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

START = "2020-07-27T00:00:00Z"
END = "2026-09-02T00:00:00Z"

ORB_WINDOW_MINUTES = 15
ATR_LEN = 14
RR_RATIO = 2.0
RISK_PER_TRADE_PCT = 1.0
STARTING_EQUITY = 10000.0
FEE_PCT_PER_SIDE = 0.0001  # 0.01%/side - same conservative estimate as the other ORB labs
MIN_STOP_ATR_FRACTION = 0.05  # safety floor - see module docstring
MEASURED_MOVE_MULTIPLE = 2.0

# Already-established reference point (see project_orb_futures_strategy_findings
# memory) - 15-min ORB, immediate entry on breakout close, no confluence filter.
BASELINE_REFERENCE = {
    "label": "15min ORB, immediate entry (no retest) - REFERENCE", "trades": 687,
    "win_rate": 40.9, "profit_factor": 1.341, "net_pnl_pct": 232.5, "max_drawdown_pct": 19.8,
}

KEY_LEVEL_MODES = ["pdh_pdl", "premarket", "measured_move"]
BOT_CONFIGS = [
    ("Bot1_Retest", "pdh_pdl"), ("Bot1_Retest", "premarket"), ("Bot1_Retest", "measured_move"),
    ("Bot2_FvgOb", "pdh_pdl"), ("Bot2_FvgOb", "premarket"), ("Bot2_FvgOb", "measured_move"),
    ("Bot3_ZoneA_FarEntry_NearTarget", None),
    ("Bot3_ZoneB_NearEntry_ContinuationTarget", None),
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
    rth = df[(df["hm"] >= "09:30") & (df["hm"] <= "16:00")]
    daily = rth.groupby("session_date").agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    prev_close = daily["close"].shift(1)
    tr = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - prev_close).abs(),
        (daily["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(ATR_LEN).mean().shift(1)


def build_prev_day_hilo(df: pd.DataFrame) -> pd.DataFrame:
    """Previous REGULAR SESSION's high/low (09:30-16:00 ET only, not extended hours)."""
    rth = df[(df["hm"] >= "09:30") & (df["hm"] <= "16:00")]
    daily = rth.groupby("session_date").agg(high=("high", "max"), low=("low", "min"))
    return pd.DataFrame({"high": daily["high"].shift(1), "low": daily["low"].shift(1)})


def _orb_end_hm(window_minutes: int) -> str:
    n_bars = window_minutes // 5
    end_minute_offset = (n_bars - 1) * 5
    hour = 9 + (30 + end_minute_offset) // 60
    minute = (30 + end_minute_offset) % 60
    return f"{hour:02d}:{minute:02d}"


ORB_END_HM = _orb_end_hm(ORB_WINDOW_MINUTES)


# ---------------------------------------------------------------------------
# Zone / key-level helpers
# ---------------------------------------------------------------------------

def find_fvg_or_ob_zone(or_bars: pd.DataFrame, direction: str):
    """FVG first (bar0 vs bar2 of the 3-bar OR window), OB fallback (last
    opposite-colored OR candle, checked most-recent-first). None if neither."""
    if len(or_bars) < 3:
        return None
    a, c = or_bars.iloc[0], or_bars.iloc[2]
    if direction == "LONG" and a["high"] < c["low"]:
        return a["high"], c["low"]
    if direction == "SHORT" and a["low"] > c["high"]:
        return c["high"], a["low"]
    for _, bar in or_bars.iloc[::-1].iterrows():
        is_down, is_up = bar["close"] < bar["open"], bar["close"] > bar["open"]
        if direction == "LONG" and is_down:
            return bar["low"], bar["high"]
        if direction == "SHORT" and is_up:
            return bar["low"], bar["high"]
    return None


def find_opposite_extreme_zone(or_bars: pd.DataFrame, direction: str):
    """The single OR bar that set the range's OPPOSITE extreme from the
    breakout direction - the demand zone (LONG) / supply zone (SHORT)."""
    if or_bars.empty:
        return None
    bar = or_bars.loc[or_bars["low"].idxmin()] if direction == "LONG" else or_bars.loc[or_bars["high"].idxmax()]
    zone_low, zone_high = bar["low"], bar["high"]
    if zone_high <= zone_low:
        return None
    return zone_low, zone_high


def resolve_key_level(direction: str, mode: str, range_high: float, range_low: float,
                       prev_high, prev_low, premarket_high, premarket_low):
    if mode == "pdh_pdl":
        level = prev_high if direction == "LONG" else prev_low
    elif mode == "premarket":
        level = premarket_high if direction == "LONG" else premarket_low
    else:  # measured_move
        width = range_high - range_low
        level = range_high + MEASURED_MOVE_MULTIPLE * width if direction == "LONG" \
            else range_low - MEASURED_MOVE_MULTIPLE * width
    if level is None or pd.isna(level):
        return None
    return level


# ---------------------------------------------------------------------------
# Entry-search functions (one per bot mechanic)
# ---------------------------------------------------------------------------

def find_retest_entry(day_df, breakout_idx, direction, range_high, range_low):
    for idx in range(breakout_idx + 1, len(day_df)):
        row = day_df.iloc[idx]
        if direction == "LONG" and row["low"] <= range_high:
            return range_high, idx
        if direction == "SHORT" and row["high"] >= range_low:
            return range_low, idx
    return None


def find_zone_touch_entry(day_df, breakout_idx, direction, zone_low, zone_high, edge: str):
    """edge='near': first touch into the zone from the breakout side.
    edge='far': price must trade all the way through to the zone's far side."""
    for idx in range(breakout_idx + 1, len(day_df)):
        row = day_df.iloc[idx]
        if direction == "LONG":
            if edge == "near" and row["low"] <= zone_high:
                return zone_high, idx
            if edge == "far" and row["low"] <= zone_low:
                return zone_low, idx
        else:
            if edge == "near" and row["high"] >= zone_low:
                return zone_low, idx
            if edge == "far" and row["high"] >= zone_high:
                return zone_high, idx
    return None


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

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


def run_backtest(df: pd.DataFrame, daily_atr: pd.Series, prev_hilo: pd.DataFrame,
                  bot_mode: str, key_level_mode, fee_pct_per_side: float, sessions_filter=None) -> dict:
    equity = peak_equity = STARTING_EQUITY
    max_dd = 0.0
    trades = []
    breakout_days = zone_days = touch_days = 0

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

        or_bars = day_df[(day_df["hm"] >= "09:30") & (day_df["hm"] <= ORB_END_HM)]
        if len(or_bars) < 3:
            continue
        range_high, range_low = or_bars["high"].max(), or_bars["low"].min()
        if range_high - range_low <= 0:
            continue

        rest = day_df[day_df["hm"] > ORB_END_HM]
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

        atr = daily_atr.get(session)
        if pd.isna(atr) or atr is None or atr <= 0:
            continue

        # ---- dispatch per bot mechanic: get (entry, entry_idx, target) ----
        result = None
        if bot_mode == "Bot1_Retest":
            entry_hit = find_retest_entry(day_df, breakout_idx, direction, range_high, range_low)
            if entry_hit is not None:
                zone_days += 1
                touch_days += 1
                entry, entry_idx = entry_hit
                prev_high = prev_hilo["high"].get(session)
                prev_low = prev_hilo["low"].get(session)
                premarket_bars = day_df[day_df["hm"] < "09:30"]
                pm_high = premarket_bars["high"].max() if not premarket_bars.empty else None
                pm_low = premarket_bars["low"].min() if not premarket_bars.empty else None
                target = resolve_key_level(direction, key_level_mode, range_high, range_low,
                                            prev_high, prev_low, pm_high, pm_low)
                if target is not None:
                    result = (entry, entry_idx, target)

        elif bot_mode == "Bot2_FvgOb":
            zone = find_fvg_or_ob_zone(or_bars, direction)
            if zone is not None:
                zone_days += 1
                zone_low, zone_high = zone
                entry_hit = find_zone_touch_entry(day_df, breakout_idx, direction, zone_low, zone_high, "near")
                if entry_hit is not None:
                    touch_days += 1
                    entry, entry_idx = entry_hit
                    prev_high = prev_hilo["high"].get(session)
                    prev_low = prev_hilo["low"].get(session)
                    premarket_bars = day_df[day_df["hm"] < "09:30"]
                    pm_high = premarket_bars["high"].max() if not premarket_bars.empty else None
                    pm_low = premarket_bars["low"].min() if not premarket_bars.empty else None
                    target = resolve_key_level(direction, key_level_mode, range_high, range_low,
                                                prev_high, prev_low, pm_high, pm_low)
                    if target is not None:
                        result = (entry, entry_idx, target)

        elif bot_mode in ("Bot3_ZoneA_FarEntry_NearTarget", "Bot3_ZoneB_NearEntry_ContinuationTarget"):
            zone = find_opposite_extreme_zone(or_bars, direction)
            if zone is not None:
                zone_days += 1
                zone_low, zone_high = zone
                if bot_mode == "Bot3_ZoneA_FarEntry_NearTarget":
                    entry_hit = find_zone_touch_entry(day_df, breakout_idx, direction, zone_low, zone_high, "far")
                    if entry_hit is not None:
                        touch_days += 1
                        entry, entry_idx = entry_hit
                        target = zone_high if direction == "LONG" else zone_low
                        result = (entry, entry_idx, target)
                else:
                    entry_hit = find_zone_touch_entry(day_df, breakout_idx, direction, zone_low, zone_high, "near")
                    if entry_hit is not None:
                        touch_days += 1
                        entry, entry_idx = entry_hit
                        target = range_high if direction == "LONG" else range_low
                        result = (entry, entry_idx, target)

        if result is None:
            continue
        entry, entry_idx, target = result

        if direction == "LONG" and target <= entry:
            continue
        if direction == "SHORT" and target >= entry:
            continue

        target_distance = abs(target - entry)
        stop_distance = target_distance / RR_RATIO
        if stop_distance < MIN_STOP_ATR_FRACTION * atr:
            continue
        stop = entry - stop_distance if direction == "LONG" else entry + stop_distance

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
        "breakout_days": breakout_days, "zone_days": zone_days, "touch_days": touch_days,
    }


def print_row(label, r):
    small = " (small sample - treat cautiously)" if 0 < r["trades"] < 30 else ""
    print(f"{label:<58} {r['trades']:>7} {r['win_rate']:>6.1f}% "
          f"{r['profit_factor']:>13.3f} {r['net_pnl_pct']:>+9.1f}% {r['max_drawdown_pct']:>7.1f}%{small}")


def config_label(bot_mode, key_level_mode):
    return f"{bot_mode} / key={key_level_mode}" if key_level_mode else bot_mode


def main():
    log.info("Fetching QQQ 5-minute bars (%s to %s) from Alpaca (IEX feed)...", START, END)
    df = fetch_bars()
    sessions_all = sorted(df["session_date"].unique())
    log.info("Got %d 5-minute bars across %d sessions.", len(df), len(sessions_all))

    premarket_bar_counts = df[df["hm"] < "09:30"].groupby("session_date").size()
    pct_sessions_with_premarket = (len(premarket_bar_counts) / len(sessions_all) * 100) if sessions_all else 0
    log.info("DATA COVERAGE CHECK: %d/%d sessions (%.1f%%) have at least one bar before 09:30 ET "
             "(avg %.1f such bars/session where present). If this is ~0%%, the 'premarket' key-level "
             "variant below will show ~0 trades - that's a data gap, not a strategy result.",
             len(premarket_bar_counts), len(sessions_all), pct_sessions_with_premarket,
             premarket_bar_counts.mean() if not premarket_bar_counts.empty else 0)

    daily_atr = build_daily_atr(df)
    prev_hilo = build_prev_day_hilo(df)

    print("\n" + "=" * 112)
    print("PART 1: FULL-SAMPLE GRID, 15-min ORB, 3 new entry mechanics x key-level variants, TP-first/2RR risk model")
    print("=" * 112)
    print(f"{'Config':<58} {'Trades':>7} {'Win%':>7} {'ProfitFactor':>13} {'Net P&L%':>10} {'MaxDD%':>8}")
    print("-" * 112)
    print_row(BASELINE_REFERENCE["label"], BASELINE_REFERENCE)
    print("-" * 112)
    full_results = {}
    for bot_mode, key_level_mode in BOT_CONFIGS:
        label = config_label(bot_mode, key_level_mode)
        log.info("Running: %s ...", label)
        r = run_backtest(df, daily_atr, prev_hilo, bot_mode, key_level_mode, FEE_PCT_PER_SIDE)
        full_results[(bot_mode, key_level_mode)] = r
        print_row(label, r)
    print("=" * 112)

    print("\nFunnel (breakout confirmed -> zone/level found -> retracement touched -> traded):")
    print("-" * 112)
    for (bot_mode, key_level_mode), r in full_results.items():
        label = config_label(bot_mode, key_level_mode)
        print(f"{label:<58} breakouts={r['breakout_days']:>4}  zone_found={r['zone_days']:>4}  "
              f"touched={r['touch_days']:>4}  traded={r['trades']:>4}")
    print("-" * 112)

    # ---- walk-forward split ----
    split_idx = len(sessions_all) // 2
    early_sessions = set(sessions_all[:split_idx])
    late_sessions = set(sessions_all[split_idx:])
    split_date = sessions_all[split_idx]

    print(f"\nPART 2: WALK-FORWARD CHECK (early: {sessions_all[0]} to {sessions_all[split_idx-1]}"
          f"  |  late: {split_date} to {sessions_all[-1]}), with fees")
    print("=" * 112)
    print(f"{'Config':<58} {'Trades':>7} {'Win%':>7} {'ProfitFactor':>13} {'Net P&L%':>10} {'MaxDD%':>8}")
    print("-" * 112)
    for bot_mode, key_level_mode in BOT_CONFIGS:
        base_label = config_label(bot_mode, key_level_mode)
        for period_name, sess_set in [("EARLY", early_sessions), ("LATE", late_sessions)]:
            r = run_backtest(df, daily_atr, prev_hilo, bot_mode, key_level_mode, FEE_PCT_PER_SIDE,
                              sessions_filter=sess_set)
            print_row(f"{base_label} - {period_name}", r)
    print("=" * 112)
    print("\nA config only deserves trust if PF > 1.0 in BOTH halves AND the full-sample trade count is large")
    print("enough to not be noise (see print_row's small-sample flag) - pooling/cross-validation rules from")
    print("project methodology memory apply here same as anywhere else.\n")


if __name__ == "__main__":
    main()
