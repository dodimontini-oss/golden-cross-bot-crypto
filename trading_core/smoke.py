"""Explicit paper-account order checks used by manual workflow runs."""
import hashlib
import os

import requests

from .execution import Alpaca, TERMINAL


def alpaca_cancel_test(base, headers, symbol, *, quantity="1", limit_price="1", broker=None):
    """Submit an intentionally unfillable paper limit order, then cancel it."""
    if base.rstrip("/") != "https://paper-api.alpaca.markets":
        raise RuntimeError("Smoke tests are restricted to Alpaca paper trading")

    run_key = "|".join((
        os.getenv("GITHUB_REPOSITORY", "local"),
        os.getenv("GITHUB_RUN_ID", "manual"),
        os.getenv("GITHUB_RUN_ATTEMPT", "1"),
        symbol,
    ))
    client_id = "smoke-" + hashlib.sha256(run_key.encode()).hexdigest()[:24]
    broker = broker or Alpaca(base, headers)
    side = "buy"
    if "/" in symbol:
        position = broker.position(symbol)
        held = float(position.get("qty", 0)) if position else 0.0
        if held >= 0.00001:
            side, quantity, limit_price = "sell", "0.00001", "1000000"

    try:
        order = broker.submit({
            "symbol": symbol,
            "qty": str(quantity),
            "side": side,
            "type": "limit",
            "limit_price": str(limit_price),
            "time_in_force": "gtc",
            "client_order_id": client_id,
        })
    except requests.exceptions.HTTPError as exc:
        response = exc.response
        detail = response.text[:300] if response is not None else str(exc)
        raise RuntimeError(f"Alpaca rejected smoke order: {detail}") from exc
    status = str(order.get("status", "")).lower()
    if status == "filled":
        raise RuntimeError(f"Smoke order unexpectedly filled: {order.get('id')}")
    if status in TERMINAL and status != "canceled":
        raise RuntimeError(f"Smoke order was not accepted: status={status}")

    canceled = order if status == "canceled" else broker.cancel(order)
    final_status = str(canceled.get("status", "")).lower()
    if final_status != "canceled":
        raise RuntimeError(f"Smoke order did not cancel cleanly: status={final_status}")

    print(
        "SMOKE_TEST_OK "
        f"broker=alpaca symbol={symbol} side={side} order_id={order.get('id')} "
        f"submitted_status={status or 'unknown'} final_status={final_status}"
    )
    return canceled
