"""
One-off diagnostic: why do MATICUSDT/ATOMUSDT/ALGOUSDT/XLMUSDT/ETCUSDT come
back with 0 bars in donchian_live_bot.py's live runs, every single cycle,
while the exact same request returns full data when run from a residential
US (California/Comcast) IP?

Hypothesis under test: the GitHub Actions runner's outbound IP is a
datacenter/cloud IP that Binance.US treats differently for a SUBSET of
symbols (possibly tied to the 2023 SEC enforcement actions naming several
tokens - including Polygon/MATIC and, in some filings, ATOM/ALGO among
others - as alleged unregistered securities; several US exchanges responded
by restricting programmatic/non-retail access to those specific tokens
without a full delisting). This prints exactly what the runner sees so the
hypothesis can be confirmed or ruled out with real evidence rather than
inferred from off-runner testing.

Not wired into any bot - pure read-only diagnostic. No credentials needed
(all public Binance.US endpoints).

Run via workflow_dispatch only.
"""

import json
import time

import requests

SYMBOLS = ["MATICUSDT", "ATOMUSDT", "ALGOUSDT", "XLMUSDT", "ETCUSDT", "BTCUSDT"]


def show(label, resp):
    print(f"--- {label} ---")
    print("status:", resp.status_code)
    interesting_headers = ["content-type", "x-mbx-used-weight", "x-mbx-used-weight-1m",
                            "cf-ray", "server", "via", "x-cache", "cf-cache-status"]
    print("headers:", {k: resp.headers.get(k) for k in interesting_headers if resp.headers.get(k)})
    body = resp.text
    print("body length:", len(body))
    print("body snippet:", body[:300])
    print()


def main():
    print("=" * 90)
    print("RUNNER NETWORK IDENTITY")
    print("=" * 90)
    try:
        r = requests.get("http://ip-api.com/json", timeout=10)
        print(json.dumps(r.json(), indent=2))
    except Exception as e:
        print("geo lookup failed:", e)
    print()

    print("=" * 90)
    print("KLINES - exact same call donchian_live_bot.py makes (limit=275)")
    print("=" * 90)
    for sym in SYMBOLS:
        try:
            r = requests.get("https://api.binance.us/api/v3/klines",
                              params={"symbol": sym, "interval": "1d", "limit": 275}, timeout=15)
            n = len(r.json()) if r.status_code == 200 else "n/a"
            print(f"{sym}: status={r.status_code} bars_returned={n}")
        except Exception as e:
            print(f"{sym}: EXCEPTION {e}")
    print()

    print("=" * 90)
    print("KLINES - small limit, full response shown")
    print("=" * 90)
    for sym in SYMBOLS:
        try:
            r = requests.get("https://api.binance.us/api/v3/klines",
                              params={"symbol": sym, "interval": "1d", "limit": 5}, timeout=15)
            show(f"klines {sym} (limit=5)", r)
        except Exception as e:
            print(f"{sym}: EXCEPTION {e}")
        time.sleep(0.2)

    print("=" * 90)
    print("EXCHANGEINFO - symbol status/permissions as seen by THIS runner")
    print("=" * 90)
    try:
        ei = requests.get("https://api.binance.us/api/v3/exchangeInfo", timeout=20)
        print("exchangeInfo top-level status:", ei.status_code)
        data = ei.json()
        by_symbol = {s["symbol"]: s for s in data.get("symbols", [])}
        for sym in SYMBOLS:
            info = by_symbol.get(sym)
            if info:
                print(f"{sym}: status={info.get('status')} permissions={info.get('permissions')} "
                      f"isSpotTradingAllowed={info.get('isSpotTradingAllowed')}")
            else:
                print(f"{sym}: NOT FOUND in exchangeInfo")
    except Exception as e:
        print("exchangeInfo EXCEPTION:", e)
    print()

    print("=" * 90)
    print("RETRY BEHAVIOR - does a second attempt a few seconds later change anything?")
    print("=" * 90)
    for sym in ["ATOMUSDT", "ALGOUSDT"]:
        for attempt in range(3):
            try:
                r = requests.get("https://api.binance.us/api/v3/klines",
                                  params={"symbol": sym, "interval": "1d", "limit": 275}, timeout=15)
                n = len(r.json()) if r.status_code == 200 else "n/a"
                print(f"{sym} attempt {attempt+1}: status={r.status_code} bars={n}")
            except Exception as e:
                print(f"{sym} attempt {attempt+1}: EXCEPTION {e}")
            time.sleep(3)


if __name__ == "__main__":
    main()
