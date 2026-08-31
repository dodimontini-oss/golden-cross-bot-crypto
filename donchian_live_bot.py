"""
Donchian breakout crypto bot (Alpaca execution) - the strategy that came
out of the 2026-08-30 multi-strategy shootout and exit-optimization work.
Intended to REPLACE the Golden Cross logic in live_bot.py, not run beside
it (both would trade the same Alpaca account and fight over positions).

Strategy, exactly as validated (see project_multi_strategy_shootout memory):
  ENTRY  long when, on the last CLOSED daily candle:
           close > (highest high of the prior 55 days) + 1.0 * ATR(14)
         AND Bitcoin itself is above its own 200-day SMA (the "BTC regime"
         filter - BTC leads the whole complex, and gating on it added
         +0.188 PF and cut drawdown).
  EXIT   whichever comes first:
           - initial stop at entry - 4.0 * ATR(14)-at-entry, or
           - a close below the 10-day low (Turtle-style channel exit).
         There is NO fixed profit target. That is the point: the fixed 2:1
         target the old bot used was capping the runners this strategy
         depends on. Replacing it roughly doubled profit factor
         (1.97 -> 3.72) at comparable drawdown, and the channel-10 variant
         had the most stable walk-forward of anything tested
         (PF 3.757 early vs 3.699 late - essentially no decay).
  RISK   2.0% of equity per trade, chosen deliberately with the user on
         2026-08-30. Backtested at 2% this returns ~29% CAGR at ~36% TRUE
         (mark-to-market) drawdown over 2019-09-23..2026-08-30, versus the
         old bot's ~3% CAGR. The drawdown is the real cost of that and was
         an explicit decision, not a default - see RISK_PER_TRADE_PCT below
         before changing it.

Universe deliberately stays at the original 17 large caps. Expanding to all
147 liquid Binance.US USDT pairs was tested and made things WORSE (late-half
walk-forward PF collapsed 3.74 -> 1.50, and uncapped concurrency pushed
drawdown to 70%); the small-cap breakouts fail more and the backtest doesn't
even model their worse slippage. Don't widen it.

Three execution realities this design has to work around, all confirmed by
direct testing against Alpaca (see live_bot.py's docstring, same account):

1. Alpaca rejects standalone stop orders for crypto - no resting stop is
   possible. This bot enforces BOTH exits itself each cycle.

2. **This strategy has no resting protective order at all.** The old bot at
   least kept a resting take-profit limit order, so a target could still
   fill while the bot was offline. A channel exit has no fixed price, so
   there is nothing to rest. If this bot stops running, open positions have
   NO protection whatsoever - no stop, no target. That is a genuine
   step down in offline safety versus the old bot and the scheduled job
   must be kept running reliably.

3. MARKET DATA COMES FROM ALPACA, NOT BINANCE.US (changed 2026-08-31).
   The strategy was backtested on Binance.US daily candles, and this bot
   originally read them live. On 2026-08-31 api.binance.us stopped
   resolving - from GitHub runners AND from a normal home connection - and
   eight consecutive scheduled runs died before managing a single position.
   With no resting stop orders at Alpaca (see 1 and 2), a data outage means
   open positions go completely unwatched, so a single-venue dependency on
   an exchange we do not even trade through was the wrong design.
   Alpaca's crypto bars are public (no auth), and Alpaca is the venue that
   actually fills these orders, so data and execution now agree. Measured
   2026-08-31 against the cached Binance history over a 12-day overlap:
   median absolute daily-close difference 0.018% (BTC), 0.037% (ETH),
   0.155% (LINK); worst case 0.55%. That is immaterial for a daily
   strategy with 4xATR stops, but it does mean live Donchian levels can sit
   a hair away from the backtested ones - a known, bounded difference, not
   an unexamined one.

4. Alpaca crypto is spot-only (no shorting), which is fine here - the
   validated strategy is long-only anyway, so unlike the Golden Cross bot
   there is no gap between what was backtested and what can be executed.

Because GitHub Actions runs are stateless, nothing is stored between runs.
The channel exit is recomputed from market data each cycle (naturally
stateless). The initial stop needs the ATR as of the ENTRY bar, which is
recovered by looking up the position's opening fill in Alpaca's own order
history and recomputing ATR as of that date - no state file needed.

Signals are taken from the last CLOSED daily candle only; the feed includes
the current in-progress day, and acting on it would mean trading a partial
bar the backtest never saw, so it is filtered out by date. Schedule this shortly after 00:00 UTC
so entries land near the next daily open, matching the backtest's
"signal on close, enter at next open" rule.

Modes:
    python donchian_live_bot.py once   - one pass over all pairs, then exit
                                          (for scheduled/GitHub Actions use).
    python donchian_live_bot.py        - loop forever, checking hourly.

Environment variables required:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
"""

import logging
import os
import sys
import time

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("crypto-donchian")

ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # paper only - never change without a deliberate decision
CRYPTO_DATA_URL = "https://data.alpaca.markets/v1beta3/crypto/us"  # market data - see the data-source note in the docstring

ALPACA_HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

GRANULARITY = "1d"
DONCHIAN_LEN = 55        # entry channel
STRENGTH_ATR = 1.0       # breakout must clear the channel by this much ATR
EXIT_CHANNEL_LEN = 10    # exit on a close below this many days' low
ATR_LEN = 14
ATR_STOP_MULT = 4.0
BTC_REGIME_LEN = 200     # BTC must be above its own SMA of this length

# 2.0% was a deliberate joint decision on 2026-08-30 to lift returns from
# ~3%/yr to ~29%/yr, accepting ~36% backtested drawdown. Raising it further
# keeps working arithmetically (3% -> ~41% CAGR at ~44% DD) but the drawdown
# becomes very hard to sit through. Do not change without re-reading the
# tradeoff table in project_multi_strategy_shootout memory.
RISK_PER_TRADE_PCT = 2.0

CANDLES_NEEDED = BTC_REGIME_LEN + ATR_LEN + 60  # enough for every indicator plus slack

PAIR_MAP = {
    "BTCUSDT": "BTC/USD", "ETHUSDT": "ETH/USD", "SOLUSDT": "SOL/USD",
    "XRPUSDT": "XRP/USD", "ADAUSDT": "ADA/USD", "DOGEUSDT": "DOGE/USD",
    "LTCUSDT": "LTC/USD", "LINKUSDT": "LINK/USD", "AVAXUSDT": "AVAX/USD",
    "DOTUSDT": "DOT/USD", "MATICUSDT": "MATIC/USD", "BCHUSDT": "BCH/USD",
    "UNIUSDT": "UNI/USD", "ATOMUSDT": "ATOM/USD", "ALGOUSDT": "ALGO/USD",
    "XLMUSDT": "XLM/USD", "ETCUSDT": "ETC/USD",
}


def alpaca_position_symbol(alpaca_symbol: str) -> str:
    """Positions/orders-by-symbol endpoints want the no-slash form."""
    return alpaca_symbol.replace("/", "")


def _get_with_retry(url, *, params=None, headers=None, timeout=15, attempts=3):
    """GitHub-hosted runners intermittently fail DNS resolution (a real
    2026-08-31 failure: 'Failed to resolve api.binance.us'). A single
    transient blip should not cost a whole cycle - and with no resting stop
    orders on Alpaca, a lost cycle means open positions go unchecked."""
    last = None
    for i in range(attempts):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last = e
            if i < attempts - 1:
                wait = 2 ** i
                log.warning("Network error on %s (attempt %d/%d): %s - retrying in %ds.",
                            url, i + 1, attempts, e.__class__.__name__, wait)
                time.sleep(wait)
    raise last


# ---------------- market data (Binance.US - matches the validated backtest) ----------------

def get_recent_candles(symbol: str, count: int) -> pd.DataFrame:
    """Returns CLOSED daily candles only, from Alpaca.

    The current (in-progress) UTC day is dropped explicitly by date rather
    than by position: the backtest only ever saw closed bars, and acting on
    a partial bar would be trading a candle the strategy was never validated
    on."""
    start = (pd.Timestamp.now('UTC').normalize() - pd.Timedelta(days=count + 120)).strftime("%Y-%m-%d")
    resp = _get_with_retry(f"{CRYPTO_DATA_URL}/bars",
                           params={"symbols": symbol, "timeframe": "1D",
                                   "start": start, "limit": 10000}, timeout=20)
    bars = resp.json().get("bars", {}).get(symbol, [])
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame([
        {"time": pd.to_datetime(b["t"], utc=True).normalize(), "open": float(b["o"]),
         "high": float(b["h"]), "low": float(b["l"]), "close": float(b["c"])}
        for b in bars
    ]).sort_values("time").reset_index(drop=True)
    today = pd.Timestamp.now('UTC').normalize()
    return df[df["time"] < today].reset_index(drop=True)


def get_live_price(symbol: str) -> float:
    resp = _get_with_retry(f"{CRYPTO_DATA_URL}/latest/trades",
                           params={"symbols": symbol}, timeout=10)
    return float(resp.json()["trades"][symbol]["p"])


def wilder_rma(series: pd.Series, length: int) -> pd.Series:
    rma = pd.Series(index=series.index, dtype=float)
    if len(series) < length:
        return rma
    rma.iloc[length - 1] = series.iloc[:length].mean()
    for i in range(length, len(series)):
        rma.iloc[i] = (rma.iloc[i - 1] * (length - 1) + series.iloc[i]) / length
    return rma


def atr_wilder(df: pd.DataFrame, length: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return wilder_rma(tr, length)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr"] = atr_wilder(df, ATR_LEN)
    # prior-N extremes, shifted so the current bar never sits inside the
    # channel it is being tested against
    df["dc_high"] = df["high"].rolling(DONCHIAN_LEN).max().shift(1)
    df["exit_low"] = df["low"].rolling(EXIT_CHANNEL_LEN).min().shift(1)
    return df


def btc_regime_is_bullish():
    """BTC above its own 200-day SMA. Fetched once per cycle and reused.

    Returns True/False, or None when the regime genuinely could not be
    determined (network failure). None is NOT the same as bearish: it blocks
    new entries but must never block exit management, which is the whole
    reason this is separated out rather than raising."""
    try:
        df = get_recent_candles("BTC/USD", CANDLES_NEEDED)
    except Exception as e:
        log.error("Could not fetch BTC data to judge the regime (%s) - entries disabled this cycle, "
                  "exits still enforced.", e.__class__.__name__)
        return None
    if len(df) < BTC_REGIME_LEN + 1:
        log.warning("Not enough BTC history (%d bars) to judge the regime - no new entries.", len(df))
        return None
    sma = df["close"].rolling(BTC_REGIME_LEN).mean().iloc[-1]
    close = df["close"].iloc[-1]
    bullish = bool(close > sma)
    log.info("BTC regime: close=%.2f vs SMA%d=%.2f -> %s",
             close, BTC_REGIME_LEN, sma, "BULLISH (entries allowed)" if bullish else "BEARISH (no new entries)")
    return bullish


# ---------------- Alpaca (execution only) ----------------

def get_account():
    resp = requests.get(f"{ALPACA_BASE_URL}/v2/account", headers=ALPACA_HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_position(alpaca_symbol: str):
    """Returns (qty, avg_entry_price); (0.0, None) when flat."""
    pos_symbol = alpaca_position_symbol(alpaca_symbol)
    resp = requests.get(f"{ALPACA_BASE_URL}/v2/positions/{pos_symbol}", headers=ALPACA_HEADERS, timeout=10)
    if resp.status_code == 404:
        return 0.0, None
    resp.raise_for_status()
    pos = resp.json()
    qty = float(pos["qty"])
    return (qty if pos["side"] == "long" else -qty), float(pos["avg_entry_price"])


def get_last_entry_time(alpaca_symbol: str):
    """When the currently-open position was opened, from Alpaca's own fill
    history. Used to recompute the ATR as of the entry bar so the initial
    stop matches the backtest without needing any stored state."""
    params = {"status": "closed", "symbols": alpaca_symbol, "direction": "desc", "limit": 50}
    resp = requests.get(f"{ALPACA_BASE_URL}/v2/orders", headers=ALPACA_HEADERS, params=params, timeout=10)
    resp.raise_for_status()
    for order in resp.json():
        if order.get("side") == "buy" and order.get("filled_at"):
            return pd.to_datetime(order["filled_at"], utc=True)
    return None


def cancel_stale_limit_orders(alpaca_symbol: str) -> int:
    """This strategy NEVER rests a limit order - the exit has no fixed price.
    So any open limit order on one of our symbols is a leftover take-profit
    from the previous Golden Cross bot, and it is actively dangerous: once
    that position closes the order is orphaned, and if a later Donchian
    trade opens on the same symbol the stale order can suddenly become
    fillable and sell the new position at the OLD bot's target price.
    Cleaned up on sight."""
    params = {"status": "open", "symbols": alpaca_symbol}
    resp = requests.get(f"{ALPACA_BASE_URL}/v2/orders", headers=ALPACA_HEADERS, params=params, timeout=10)
    resp.raise_for_status()
    cancelled = 0
    for order in resp.json():
        if order.get("type") == "limit":
            requests.delete(f"{ALPACA_BASE_URL}/v2/orders/{order['id']}", headers=ALPACA_HEADERS, timeout=10)
            log.warning("[%s] Cancelled a stale resting limit order (id=%s, price=%s) left over from the "
                        "previous strategy - this bot uses no resting orders.",
                        alpaca_symbol, order["id"], order.get("limit_price"))
            cancelled += 1
    return cancelled


def market_order(alpaca_symbol: str, qty: float, side: str) -> dict:
    body = {"symbol": alpaca_symbol, "qty": str(round(abs(qty), 6)), "side": side,
            "type": "market", "time_in_force": "gtc"}
    resp = requests.post(f"{ALPACA_BASE_URL}/v2/orders", headers=ALPACA_HEADERS, json=body, timeout=15)
    if resp.status_code >= 400:
        log.error("[%s] Alpaca rejected the %s order (status %d): %s",
                  alpaca_symbol, side, resp.status_code, resp.text)
    resp.raise_for_status()
    return resp.json()


# ---------------- per-pair logic ----------------

def manage_open_position(binance_symbol, alpaca_symbol, qty, avg_entry, df):
    """Enforces both exits. Stop is checked against the LIVE price (the
    backtest checked it intrabar against the bar's low); the channel exit is
    checked on the last closed candle's close, exactly as backtested. Stop
    is evaluated first, matching the backtest's conservative assumption that
    the stop fills first when a bar could have hit both."""
    last = df.iloc[-1]

    entry_time = get_last_entry_time(alpaca_symbol)
    stop_price = None
    if entry_time is not None:
        at_entry = df[df["time"] <= entry_time]
        if len(at_entry) and pd.notna(at_entry.iloc[-1]["atr"]):
            stop_price = avg_entry - at_entry.iloc[-1]["atr"] * ATR_STOP_MULT
    if stop_price is None:
        # Fall back to the current ATR rather than running with no stop at
        # all. Slightly different from the backtest (which fixes the stop at
        # entry), but an unprotected position is the worse failure.
        if pd.notna(last["atr"]):
            stop_price = avg_entry - last["atr"] * ATR_STOP_MULT
            log.warning("[%s] Could not recover the entry-bar ATR - using the current ATR for the stop.", binance_symbol)
        else:
            log.warning("[%s] No ATR available - cannot evaluate the stop this cycle.", binance_symbol)
            return

    live_price = get_live_price(alpaca_symbol)
    if live_price <= stop_price:
        log.info("[%s] STOP breached (price=%.6f stop=%.6f) - closing.", binance_symbol, live_price, stop_price)
        market_order(alpaca_symbol, qty, "sell")
        return

    exit_low = last["exit_low"]
    if pd.notna(exit_low) and last["close"] < exit_low:
        log.info("[%s] CHANNEL exit (last close=%.6f < %d-day low=%.6f) - closing.",
                 binance_symbol, last["close"], EXIT_CHANNEL_LEN, exit_low)
        market_order(alpaca_symbol, qty, "sell")
        return

    log.info("[%s] Holding %.6f units. price=%.6f stop=%.6f %dd-low=%.6f",
             binance_symbol, qty, live_price, stop_price, EXIT_CHANNEL_LEN,
             exit_low if pd.notna(exit_low) else float("nan"))


def try_entry(binance_symbol, alpaca_symbol, df, equity, available_cash):
    last = df.iloc[-1]
    if pd.isna(last["dc_high"]) or pd.isna(last["atr"]) or last["atr"] <= 0:
        log.info("[%s] Indicators not warmed up yet. No action.", binance_symbol)
        return

    trigger = last["dc_high"] + STRENGTH_ATR * last["atr"]
    if last["close"] <= trigger:
        log.info("[%s] No breakout (close=%.6f, needs > %.6f = %dd-high %.6f + %.1fxATR). No action.",
                 binance_symbol, last["close"], trigger, DONCHIAN_LEN, last["dc_high"], STRENGTH_ATR)
        return

    stop_distance = last["atr"] * ATR_STOP_MULT
    risk_amount = equity * RISK_PER_TRADE_PCT / 100
    qty = risk_amount / stop_distance

    # Cap the position by cash actually available. Learned the hard way on
    # the ORB bot, which had a live order rejected for insufficient buying
    # power because pure risk-based sizing ignored what the account could
    # afford. This can only ever shrink a position, never grow it.
    live_price = get_live_price(alpaca_symbol)
    max_qty = (available_cash * 0.95) / live_price if live_price > 0 else 0
    if max_qty <= 0:
        log.warning("[%s] Breakout signalled but no cash available - skipping.", binance_symbol)
        return
    if qty > max_qty:
        log.info("[%s] Sizing capped by available cash: %.6f -> %.6f units.", binance_symbol, qty, max_qty)
        qty = max_qty

    stop_price = last["close"] - stop_distance
    log.info("[%s] BREAKOUT confirmed (close=%.6f > %.6f) - buying %.6f units, initial stop %.6f, "
             "exit on a close below the %dd low.",
             binance_symbol, last["close"], trigger, qty, stop_price, EXIT_CHANNEL_LEN)
    market_order(alpaca_symbol, qty, "buy")


def check_and_trade(binance_symbol, alpaca_symbol, allow_entries, equity, available_cash):
    cancel_stale_limit_orders(alpaca_symbol)

    df = get_recent_candles(alpaca_symbol, CANDLES_NEEDED)
    if len(df) < DONCHIAN_LEN + ATR_LEN + 2:
        log.warning("[%s] Not enough candle history (%d bars).", binance_symbol, len(df))
        return
    df = add_indicators(df)

    qty, avg_entry = get_position(alpaca_symbol)
    if qty != 0:
        manage_open_position(binance_symbol, alpaca_symbol, qty, avg_entry, df)
        return

    if not allow_entries:
        log.info("[%s] Flat, entries not permitted this cycle. No action.", binance_symbol)
        return

    try_entry(binance_symbol, alpaca_symbol, df, equity, available_cash)


def check_all_pairs():
    """Exits are the priority. A failure fetching the account or the BTC
    regime disables NEW ENTRIES for this cycle but must still let every open
    position be checked against its stop and channel exit - with no resting
    protective orders at Alpaca, a cycle that dies early is a cycle where
    nothing is protected. (Learned from a real 2026-08-31 run that aborted on
    a transient DNS failure before managing any position.)"""
    equity = available_cash = None
    try:
        account = get_account()
        equity = float(account["equity"])
        # Spot crypto buys settle against cash, so cash - not equity - is what
        # actually constrains a new position once others are already open.
        available_cash = float(account.get("non_marginable_buying_power") or account.get("cash") or 0.0)
        log.info("Account equity $%.2f | cash available for new positions $%.2f | risk per trade %.1f%%",
                 equity, available_cash, RISK_PER_TRADE_PCT)
    except Exception as e:
        log.error("Could not read the Alpaca account (%s) - entries disabled this cycle, "
                  "exits still enforced.", e.__class__.__name__)

    btc_bullish = btc_regime_is_bullish()
    allow_entries = (btc_bullish is True) and (equity is not None)
    if btc_bullish is False:
        log.info("BTC regime is bearish - no new entries this cycle.")
    if not allow_entries:
        log.info("Entries disabled this cycle; open positions will still be managed.")

    for binance_symbol, alpaca_symbol in PAIR_MAP.items():
        try:
            check_and_trade(binance_symbol, alpaca_symbol, allow_entries, equity, available_cash)
            if allow_entries:
                # refresh cash as positions consume it during this pass
                try:
                    available_cash = float(get_account().get("non_marginable_buying_power") or 0.0)
                except Exception:
                    log.warning("Could not refresh available cash - disabling further entries this cycle.")
                    allow_entries = False
        except requests.exceptions.HTTPError as e:
            log.warning("[%s] Skipped - %s", binance_symbol, e)
        except Exception:
            log.exception("[%s] Error during check - skipping this pair this cycle.", binance_symbol)


def run_once():
    log.info("Donchian breakout crypto bot - single pass across %d pairs.", len(PAIR_MAP))
    check_all_pairs()


def run_live(poll_interval_seconds: int = 3600):
    log.info("Donchian breakout crypto bot starting - %d pairs, polling every %ds.",
             len(PAIR_MAP), poll_interval_seconds)
    while True:
        try:
            check_all_pairs()
        except Exception:
            log.exception("Unexpected error in main loop - will retry next poll.")
        time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "live"
    if mode == "once":
        run_once()
    else:
        run_live()
