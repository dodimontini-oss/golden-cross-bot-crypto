"""
One-off diagnostic: does the account's existing Alpaca API key (already
stored as ALPACA_API_KEY/ALPACA_SECRET_KEY for the crypto bot) also have
access to Alpaca's stock market data API? If so, how far back does 5-minute
QQQ history actually go? Yahoo Finance caps 5m/15m/30m intraday data at 60
days - if Alpaca's free IEX-sourced bars go back further, that's a much
better data source for backtesting an ORB futures-proxy strategy on QQQ (as
a stand-in for NQ, since CME futures data itself isn't freely available).

Not a live-trading script - read-only market data query only.

Run: python check_alpaca_stock_data.py

Environment variables required:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
"""

import logging
import os

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("check-alpaca-stock-data")

ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
DATA_URL = "https://data.alpaca.markets/v2/stocks/QQQ/bars"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}


def main():
    # try a wide date range - if the account can see stock data at all, and
    # how far back 5-minute bars actually go.
    params = {
        "timeframe": "5Min",
        "start": "2015-01-01T00:00:00Z",
        "end": "2026-08-27T00:00:00Z",
        "limit": 10000,
        "feed": "iex",
    }
    resp = requests.get(DATA_URL, headers=HEADERS, params=params, timeout=30)
    log.info("Status: %d", resp.status_code)
    log.info("Body (first 2000 chars): %s", resp.text[:2000])

    if resp.status_code != 200:
        return

    data = resp.json()
    bars = data.get("bars", [])
    log.info("Bars returned this page: %d", len(bars))
    if bars:
        log.info("First bar: %s", bars[0])
        log.info("Last bar: %s", bars[-1])
    log.info("next_page_token present: %s", bool(data.get("next_page_token")))


if __name__ == "__main__":
    main()
