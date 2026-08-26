"""
One-off trade history report for the live Golden Cross crypto bot. Not part
of the bot's own logic - just reads Alpaca's own position and order records
and prints a summary. Read-only: no orders are placed.

Alpaca doesn't group fills into a single "closed trade" record the way OANDA
does (see forex_screener/golden_cross_trade_history.py) - each buy and each
sell is its own filled order. This reports filled orders in chronological
order per symbol so entries and exits can be read side by side, plus current
open positions.

Environment variables required:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
"""

import os

import requests

ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}


def get_account() -> dict:
    resp = requests.get(f"{ALPACA_BASE_URL}/v2/account", headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_positions() -> list:
    resp = requests.get(f"{ALPACA_BASE_URL}/v2/positions", headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_all_filled_orders() -> list:
    """Newest-first from Alpaca; re-sorted chronologically below."""
    params = {"status": "all", "limit": 500, "direction": "desc"}
    resp = requests.get(f"{ALPACA_BASE_URL}/v2/orders", headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    orders = resp.json()
    return [o for o in orders if o.get("filled_at")]


def run():
    account = get_account()
    equity = float(account["equity"])
    currency = "USD"

    positions = get_positions()
    orders = get_all_filled_orders()
    orders.sort(key=lambda o: o["filled_at"])

    print(f"\nAlpaca account summary: equity={equity:.2f} {currency}\n")

    print(f"=== OPEN POSITIONS ({len(positions)}) ===")
    if not positions:
        print("  (none)")
    for p in positions:
        symbol = p["symbol"]
        qty = float(p["qty"])
        side = p["side"]
        entry = float(p["avg_entry_price"])
        current = float(p["current_price"])
        upl = float(p["unrealized_pl"])
        upl_pct = float(p["unrealized_plpc"]) * 100
        print(f"  [{symbol}] {side.upper():<5} qty={qty} @ {entry:.4f} -> current {current:.4f} "
              f"| unrealized P/L={upl:+.2f} {currency} ({upl_pct:+.1f}%)")

    print(f"\n=== FILLED ORDERS ({len(orders)}) - chronological ===")
    if not orders:
        print("  (none)")
    buys = sells = 0
    for o in orders:
        symbol = o["symbol"]
        side = o["side"]
        qty = float(o["filled_qty"])
        avg_price = float(o["filled_avg_price"]) if o.get("filled_avg_price") else None
        order_type = o["type"]
        filled_at = o["filled_at"][:19] if o.get("filled_at") else "?"
        if side == "buy":
            buys += 1
        else:
            sells += 1
        price_str = f"{avg_price:.4f}" if avg_price is not None else "?"
        print(f"  [{symbol}] {side.upper():<4} {order_type:<6} qty={qty} @ {price_str} "
              f"filled {filled_at}")

    print(f"\n=== SUMMARY ===")
    print(f"Filled orders: {len(orders)}  (buys: {buys}, sells: {sells})")
    print(f"Open positions: {len(positions)}")
    total_unrealized = sum(float(p["unrealized_pl"]) for p in positions)
    print(f"Current total unrealized P/L across open positions: {total_unrealized:+.2f} {currency}")
    print("Note: Alpaca reports fills, not matched round-trip trades - to see "
          "realized P/L per closed trade, pair each symbol's buy/sell fills above "
          "by eye (the bot is long-only and pyramiding=0, so at most one open "
          "position per symbol at a time, making buy->sell pairs unambiguous).")
    print()


if __name__ == "__main__":
    run()
