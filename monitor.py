"""Headless monitoring pass for scheduling (cron / Task Scheduler / a server).

Checks every ticker for new SEC filings, extracts management statements, runs a consistency review
and sends alerts (email / webhook) using the rules saved in the app's "Alerts & settings" tab.

Usage:
  export ANTHROPIC_API_KEY=...  SEC_USER_AGENT="Jane Doe jane@fund.com"  SMTP_PASSWORD=...
  python monitor.py                 # all tickers in the saved list
  python monitor.py MSFT NVDA       # specific tickers
Reports for new reviews are also written to ./reports/.
"""
import os
import sys
from pathlib import Path

from agent import MODELS, ConsistencyAgent
from alerts import load_settings, notify
from edgar import EdgarClient
from pipeline import Store, monitor_ticker, parse_tickers, report_markdown


def main() -> None:
    store = Store()
    tickers = parse_tickers(" ".join(sys.argv[1:])) or store.watchlist()
    if not tickers:
        sys.exit("No tickers given and the saved list is empty.")
    settings = load_settings()
    agent = ConsistencyAgent(os.environ["ANTHROPIC_API_KEY"], os.getenv("CLAUDE_MODEL", MODELS[0]))
    edgar = EdgarClient(os.environ["SEC_USER_AGENT"])
    out = Path(__file__).parent / "reports"
    out.mkdir(exist_ok=True)
    for t in tickers:
        print(f"== {t}")
        try:
            a = monitor_ticker(store, edgar, agent, t, log=lambda m: print("  ", m))
        except Exception as e:
            print(f"   error: {e}")
            continue
        if not a:
            print("   no new filings")
            continue
        data = store.load(t)
        report = report_markdown(data, a)
        path = out / f"{t}_{a['current_label'][:10]}.md"
        path.write_text(report)
        r = a["review"]
        print(f"   score {r['consistency_score']}/100, red flags: {len(r.get('red_flags', []))} -> {path}")
        notify(data, a, settings, report, log=lambda m: print("  ", m))


if __name__ == "__main__":
    main()
