"""
Golden Cross crypto bot (Alpaca execution) - Daily-timeframe 50/200 SMA
trend-following. Uses Binance.US for market data (matching exactly what
was backtested) and Alpaca paper trading for order execution, since
Binance blocks US-based automated trading entirely.

Two things confirmed by direct testing that shape this design:

1. Alpaca rejects standalone stop orders for crypto entirely - no resting
   stop-loss is possible. Take-profit uses a resting LIMIT order (placed
   right after entry, protected even if this bot goes offline). The
   stop-loss side is NOT protected between runs - this bot actively checks
   price each cycle and market-sells if the stop level is breached. If
   this bot stops running, open positions have zero stop-loss protection
   until it resumes. Keep the scheduled job running reliably.

   The stop level itself is never stored separately - it's reconstructed
   each cycle from the position's avg_entry_price (tracked by Alpaca) and
   the resting take-profit limit order's price (also tracked by Alpaca),
   using the known 2:1 reward:risk relationship. This avoids needing any
   separate state file, which matters because GitHub Actions runs are
   stateless between invocations.

2. Alpaca crypto is spot-only - no short selling. Short signals are
   logged and skipped, not executed. This means live behavior is
   structurally a long-only variant of the strategy that was backtested
   with both directions - re-check the long-only-specific backtest result
   before trusting this live, don't assume it performs proportionally.

Modes:
    python live_bot.py once   - checks all pairs once, then exits (for
                                 scheduled execution, e.g. GitHub Actions).
    python live_bot.py        - runs forever, checking every 5 minutes.

Environment variables required:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from trading_core import execution as safety

import logging
import os
import sys
import time

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("crypto-golden-cross")

ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # paper only - never change without a deliberate decision

BINANCE_BASE_URL = "https://api.binance.us"  # data source only - matches exactly what was backtested

ALPACA_HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

GRANULARITY = "1d"
FAST_LEN = 50
SLOW_LEN = 200
ATR_LEN = 14
ATR_STOP_MULT = 4.0  # widened from 2.0 on 2026-08-26 - walk-forward validated improvement, see crypto_bot/stop_distance_sweep_lab.py
RR_RATIO = 2.0
RISK_PER_TRADE_PCT = 1.0

# Binance.US symbol -> Alpaca symbol. Not every pair we backtested is
# necessarily listed on Alpaca - unsupported ones are skipped automatically
# at runtime based on real API responses, not a guess made in advance.
PAIR_MAP = {
    "BTCUSDT": "BTC/USD", "ETHUSDT": "ETH/USD", "SOLUSDT": "SOL/USD",
    "XRPUSDT": "XRP/USD", "ADAUSDT": "ADA/USD", "DOGEUSDT": "DOGE/USD",
    "LTCUSDT": "LTC/USD", "LINKUSDT": "LINK/USD", "AVAXUSDT": "AVAX/USD",
    "DOTUSDT": "DOT/USD", "MATICUSDT": "MATIC/USD", "BCHUSDT": "BCH/USD",
    "UNIUSDT": "UNI/USD", "ATOMUSDT": "ATOM/USD", "ALGOUSDT": "ALGO/USD",
    "XLMUSDT": "XLM/USD", "ETCUSDT": "ETC/USD",
}


def alpaca_position_symbol(alpaca_symbol: str) -> str:
    """Positions/orders-by-symbol endpoints want the no-slash form, e.g.
    'BTCUSD' not 'BTC/USD' - confirmed via direct testing."""
    return alpaca_symbol.replace("/", "")


# ---------------- Binance.US - data only, matches the validated backtest ----------------

def get_recent_candles(binance_symbol: str, count: int) -> pd.DataFrame:
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {"symbol": binance_symbol, "interval": GRANULARITY, "limit": count}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    rows = [
        {"time": pd.to_datetime(r[0], unit="ms", utc=True), "open": float(r[1]), "high": float(r[2]),
         "low": float(r[3]), "close": float(r[4])}
        for r in resp.json()
    ]
    return pd.DataFrame(rows)


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


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
    df["fast_ma"] = sma(df["close"], FAST_LEN)
    df["slow_ma"] = sma(df["close"], SLOW_LEN)
    df["atr"] = atr_wilder(df, ATR_LEN)
    return df


def find_crossovers(df: pd.DataFrame):
    for i in range(1, len(df)):
        prev, curr = df.iloc[i - 1], df.iloc[i]
        if pd.isna(prev["slow_ma"]) or pd.isna(curr["slow_ma"]):
            continue
        if prev["fast_ma"] <= prev["slow_ma"] and curr["fast_ma"] > curr["slow_ma"]:
            yield i, "LONG"
        elif prev["fast_ma"] >= prev["slow_ma"] and curr["fast_ma"] < curr["slow_ma"]:
            yield i, "SHORT"


# ---------------- Alpaca - execution only ----------------

def get_account_equity() -> float:
    resp = requests.get(f"{ALPACA_BASE_URL}/v2/account", headers=ALPACA_HEADERS, timeout=10)
    resp.raise_for_status()
    return float(resp.json()["equity"])


def get_position(alpaca_symbol: str):
    """Returns (qty, avg_entry_price). qty is signed; (0, None) if flat."""
    pos_symbol = alpaca_position_symbol(alpaca_symbol)
    resp = requests.get(f"{ALPACA_BASE_URL}/v2/positions/{pos_symbol}", headers=ALPACA_HEADERS, timeout=10)
    if resp.status_code == 404:
        return 0.0, None
    resp.raise_for_status()
    pos = resp.json()
    qty = float(pos["qty"])
    signed_qty = qty if pos["side"] == "long" else -qty
    return signed_qty, float(pos["avg_entry_price"])


def get_open_limit_order(alpaca_symbol: str):
    """Returns (limit_price, order_id) for the resting take-profit order on
    this symbol, or (None, None) if there isn't one."""
    params = {"status": "open", "symbols": alpaca_symbol}
    resp = requests.get(f"{ALPACA_BASE_URL}/v2/orders", headers=ALPACA_HEADERS, params=params, timeout=10)
    resp.raise_for_status()
    for order in resp.json():
        if order["type"] == "limit" and order["side"] == "sell":
            return float(order["limit_price"]), order["id"]
    return None, None


def market_order(alpaca_symbol: str, qty: float, side: str) -> dict:
    body = {"symbol": alpaca_symbol, "qty": str(round(abs(qty), 6)), "side": side,
            "type": "market", "time_in_force": "gtc"}
    resp = requests.post(f"{ALPACA_BASE_URL}/v2/orders", headers=ALPACA_HEADERS, json=body, timeout=15)
    resp.raise_for_status()
    return resp.json()


def limit_order(alpaca_symbol: str, qty: float, side: str, price: float) -> dict:
    broker=safety.Alpaca(ALPACA_BASE_URL,ALPACA_HEADERS)
    entry=broker.latest_entry(alpaca_symbol)
    if not entry:
        raise RuntimeError("Missing entry identity for take-profit")
    # Alpaca permits sub-dollar crypto limit precision beyond cents.
    price=round(safety.positive(price), 6 if price<1 else 2)
    return broker.submit({"symbol":alpaca_symbol,"qty":safety.quantity(qty),"side":side,
        "type":"limit","limit_price":str(price),"time_in_force":"gtc",
        "client_order_id":safety.signal_id("gc-tp",alpaca_symbol,entry.get("filled_at") or entry["submitted_at"])})


def cancel_order(order_id: str):
    requests.delete(f"{ALPACA_BASE_URL}/v2/orders/{order_id}", headers=ALPACA_HEADERS, timeout=10)


# ---------------- Core check-and-trade, per pair ----------------

def check_and_trade(binance_symbol: str, alpaca_symbol: str):
    broker = safety.Alpaca(ALPACA_BASE_URL, ALPACA_HEADERS)
    qty, avg_entry = get_position(alpaca_symbol)
    if qty:
        entry = broker.latest_entry(alpaca_symbol)
        distance = safety.order_risk(entry or {})
        target, target_id = get_open_limit_order(alpaca_symbol)
        if distance is None and target is not None:
            distance = safety.positive(target-avg_entry)/RR_RATIO
        if distance is None:
            # Fail visibly; do not pretend the position is protected. The legacy
            # migration manifest must supply its ORIGINAL stop before deployment.
            import json
            migrations = json.loads(os.environ.get("LEGACY_CRYPTO_STOPS", "{}"))
            if alpaca_symbol not in migrations:
                raise RuntimeError(f"Unprotected legacy {alpaca_symbol}: original stop required in LEGACY_CRYPTO_STOPS")
            distance = safety.positive(avg_entry-float(migrations[alpaca_symbol]))
        current = safety.positive(get_recent_candles(binance_symbol,2).iloc[-1]["close"])
        if current <= avg_entry-distance:
            broker.close(alpaca_symbol)
            return
        if target is None and not broker.orders(alpaca_symbol,"open"):
            # Reconcile settled balance on each run, including a previously delayed fill.
            limit_order(alpaca_symbol, qty, "sell", avg_entry+distance*RR_RATIO)
        return
    if broker.orders(alpaca_symbol,"open"):
        return
    df = get_recent_candles(binance_symbol,SLOW_LEN+ATR_LEN+20)
    today = pd.Timestamp.now(tz="UTC").normalize()
    df = df[df.time < today].reset_index(drop=True)
    if len(df)<SLOW_LEN+2 or df.iloc[-1].time != today-pd.Timedelta(days=1):
        raise RuntimeError("Insufficient or stale daily data")
    df=add_indicators(df)
    signals=list(find_crossovers(df))
    if not signals or signals[-1] != (len(df)-1,"LONG"):
        return
    row=df.iloc[-1]
    if any(o.get("side")=="buy" and pd.Timestamp(o["submitted_at"])>=today for o in broker.orders(alpaca_symbol)):
        return
    account=broker.get("/v2/account")
    distance=safety.positive(row.atr)*ATR_STOP_MULT
    price=safety.positive(get_recent_candles(binance_symbol,2).iloc[-1].close)
    cash=float(account.get("non_marginable_buying_power") or 0)
    trade_qty=min(float(account["equity"])*RISK_PER_TRADE_PCT/100/distance,max(0,cash)*.95/price)
    if trade_qty<=0:
        return
    broker.submit({"symbol":alpaca_symbol,"qty":safety.quantity(trade_qty),"side":"buy",
                   "type":"market","time_in_force":"gtc",
                   "client_order_id":safety.signal_id("gc",alpaca_symbol,row.time,distance)})
    # The next reconciliation manages the filled balance even if the entry is delayed.
    actual, average=get_position(alpaca_symbol)
    if actual>0 and not broker.orders(alpaca_symbol,"open"):
        limit_order(alpaca_symbol,actual,"sell",average+distance*RR_RATIO)


def check_all_pairs():
    errors = []
    for binance_symbol, alpaca_symbol in PAIR_MAP.items():
        try:
            check_and_trade(binance_symbol, alpaca_symbol)
        except requests.exceptions.HTTPError as e:
            errors.append(binance_symbol)
            log.warning("[%s] Skipped - %s", binance_symbol, e)
        except Exception:
            errors.append(binance_symbol)
            log.exception("[%s] Error during check - skipping this pair this cycle.", binance_symbol)
    if errors:
        raise RuntimeError(f"Incomplete protection cycle: {errors}")


def run_once():
    log.info("Golden Cross crypto bot - single check across %d pairs.", len(PAIR_MAP))
    check_all_pairs()


def run_live(poll_interval_seconds: int = 300):
    log.info("Golden Cross crypto bot starting (live) - %d pairs, polling every %ds.",
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
