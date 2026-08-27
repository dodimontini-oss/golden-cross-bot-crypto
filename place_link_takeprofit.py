"""
One-off: place the missing take-profit limit order for the live LINK/USD
position, which was left with zero exit protection by the bug fixed in
live_bot.py (commit 64a6582) - the original limit sell was placed for the
pre-fee-deduction requested qty, got rejected by Alpaca as insufficient
balance, and the rejection was silently swallowed by check_all_pairs()'s
generic exception handler. The position has had no resting order (and,
because of it, no virtual stop-loss check either - that branch is gated on
finding the resting order first) since it opened.

Reconstructs the EXACT target price live_bot.py would have computed at
entry time - not a re-derived estimate - using ATR_STOP_MULT=2.0 (what was
actually in effect before the 2026-08-26 widen to 4.0; this position was
already sized under the old value, so using 4.0 now would be inconsistent
with how much risk was actually taken):

    target = avg_entry_price + stop_distance * RR_RATIO
    stop_distance = risk_amount / original_requested_qty   (backed out from
        the bot's own sizing formula: trade_qty = risk_amount / stop_distance)
    risk_amount = equity_at_entry * RISK_PER_TRADE_PCT / 100
    equity_at_entry = current_equity - current_unrealized_pl_on_LINK
        (exact, not approximate: no other fills have happened on this
        account since the LINK entry - confirmed via
        golden_cross_crypto_trade_history.py - so equity has only moved by
        LINK's own unrealized P/L since then)

original_requested_qty is read from the actual entry order on Alpaca (not
hardcoded), so this is self-verifying against the broker's own record.

Places the limit sell for the ACTUAL current held qty (matches the
live_bot.py fix), not the original requested qty.

Run: python place_link_takeprofit.py

Environment variables required:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
"""

import logging
import os

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("place-link-tp")

ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

RR_RATIO = 2.0
RISK_PER_TRADE_PCT = 1.0


def main():
    resp = requests.get(f"{ALPACA_BASE_URL}/v2/account", headers=HEADERS, timeout=10)
    resp.raise_for_status()
    equity_now = float(resp.json()["equity"])
    log.info("Current account equity: %.2f USD", equity_now)

    resp = requests.get(f"{ALPACA_BASE_URL}/v2/positions/LINKUSD", headers=HEADERS, timeout=10)
    resp.raise_for_status()
    pos = resp.json()
    actual_qty = float(pos["qty"])
    avg_entry_price = float(pos["avg_entry_price"])
    unrealized_pl = float(pos["unrealized_pl"])
    log.info("LINK position: qty=%.6f avg_entry=%.4f unrealized_pl=%.2f", actual_qty, avg_entry_price, unrealized_pl)

    if actual_qty <= 0:
        log.error("LINK position shows qty=%.6f - nothing to protect, aborting.", actual_qty)
        return

    resp = requests.get(f"{ALPACA_BASE_URL}/v2/orders", headers=HEADERS,
                         params={"status": "open", "symbols": "LINK/USD"}, timeout=10)
    resp.raise_for_status()
    existing = resp.json()
    if existing:
        log.warning("LINK/USD already has %d open order(s) - not placing a duplicate. Details: %s",
                     len(existing), existing)
        return

    resp = requests.get(f"{ALPACA_BASE_URL}/v2/orders", headers=HEADERS,
                         params={"status": "closed", "symbols": "LINK/USD", "direction": "asc", "limit": 10},
                         timeout=10)
    resp.raise_for_status()
    buy_orders = [o for o in resp.json() if o["side"] == "buy" and o["type"] == "market"]
    if not buy_orders:
        log.error("Could not find the original LINK buy order on Alpaca - aborting, can't reconstruct target safely.")
        return
    original_requested_qty = float(buy_orders[0]["qty"])
    log.info("Original requested qty (from the entry order itself, pre-fee-deduction): %.6f", original_requested_qty)

    equity_at_entry = equity_now - unrealized_pl
    risk_amount = equity_at_entry * RISK_PER_TRADE_PCT / 100
    stop_distance = risk_amount / original_requested_qty
    target = avg_entry_price + stop_distance * RR_RATIO
    log.info("Reconstructed: equity_at_entry=%.2f risk_amount=%.2f stop_distance=%.4f target=%.4f",
              equity_at_entry, risk_amount, stop_distance, target)

    body = {"symbol": "LINK/USD", "qty": str(round(actual_qty, 6)), "side": "sell",
            "type": "limit", "limit_price": str(round(target, 2)), "time_in_force": "gtc"}
    resp = requests.post(f"{ALPACA_BASE_URL}/v2/orders", headers=HEADERS, json=body, timeout=15)
    resp.raise_for_status()
    order = resp.json()
    log.info("Placed take-profit limit sell: qty=%s limit_price=%s order_id=%s",
              body["qty"], body["limit_price"], order["id"])


if __name__ == "__main__":
    main()
