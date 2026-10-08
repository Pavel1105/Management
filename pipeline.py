"""Local JSON storage + the monitoring pipeline shared by the Streamlit app and the CLI."""
from __future__ import annotations

import hashlib
import re
import json
from datetime import datetime, timezone
from pathlib import Path

from agent import ConsistencyAgent
from edgar import EdgarClient

DATA_DIR = Path(__file__).parent / "data"
FORM_LABELS = {"8-K": "Earnings release (8-K)", "10-Q": "10-Q MD&A", "10-K": "10-K MD&A"}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------- storage
class Store:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str) -> Path:
        return self.dir / f"{ticker.upper()}.json"

    def load(self, ticker: str) -> dict:
        p = self._path(ticker)
        if p.exists():
            return json.loads(p.read_text())
        return {"ticker": ticker.upper(), "company": ticker.upper(), "cik": None, "docs": {}, "analyses": []}

    def save(self, data: dict) -> None:
        self._path(data["ticker"]).write_text(json.dumps(data, indent=1, ensure_ascii=False))

    def tickers(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.json") if p.stem not in ("watchlist", "settings", "alerts_log"))

    def watchlist(self) -> list[str]:
        p = self.dir / "watchlist.json"
        return json.loads(p.read_text()) if p.exists() else []

    def set_watchlist(self, tickers: list[str]) -> None:
        (self.dir / "watchlist.json").write_text(json.dumps(sorted(set(t.upper() for t in tickers))))


def add_manual_doc(data: dict, label: str, doc_type: str, date: str, text: str, url: str = "") -> str:
    doc_id = "manual-" + hashlib.sha1((label + date + text[:500]).encode()).hexdigest()[:10]
    data["docs"][doc_id] = {
        "id": doc_id, "source": "Manual", "doc_type": doc_type, "label": label, "date": date,
        "url": url, "text": text, "extraction": None, "added": now(),
    }
    return doc_id


def parse_tickers(text: str) -> list[str]:
    """'aapl, MSFT nvda;tsla' -> ['AAPL', 'MSFT', 'NVDA', 'TSLA'] (order kept, duplicates removed)."""
    seen = []
    for t in re.split(r"[\s,;]+", text.upper()):
        t = t.strip().lstrip("$")
        if t and t not in seen:
            seen.append(t)
    return seen


def portfolio_row(data: dict) -> dict:
    runs = data.get("analyses", [])
    last = runs[-1] if runs else None
    prev = runs[-2] if len(runs) > 1 else None
    score = last["review"]["consistency_score"] if last else None
    findings = last["review"].get("findings", []) if last else []
    return {
        "Ticker": data["ticker"], "Company": data["company"],
        "Documents": len(data["docs"]),
        "Latest document": max((d["date"] for d in data["docs"].values()), default=None),
        "Score": score,
        "Change": (score - prev["review"]["consistency_score"]) if (last and prev) else None,
        "High-severity issues": sum(1 for f in findings if f.get("severity") == "high"
                                    and f["verdict"] not in ("consistent", "delivered")),
        "Red flags": len(last["review"].get("red_flags", [])) if last else None,
        "Last review": last["run_at"] if last else None,
    }


def known_topics(data: dict) -> list[str]:
    return [s["topic"] for d in data["docs"].values() if d.get("extraction")
            for s in d["extraction"].get("statements", [])]


# ---------------------------------------------------------------- pipeline steps
def sync_sec(data: dict, edgar: EdgarClient, forms=("8-K", "10-Q", "10-K"), limit: int = 8,
             earnings_only: bool = True, log=print) -> list[str]:
    """Download filings not yet stored. Returns the new doc ids."""
    if not data.get("cik"):
        data["cik"], data["company"] = edgar.lookup(data["ticker"])
    new_ids = []
    for f in edgar.list_filings(data["cik"], forms, limit, earnings_only):
        doc_id = f["accession"]
        if doc_id in data["docs"]:
            continue
        log(f"Downloading {f['form']} filed {f['date']}…")
        try:
            text, url = edgar.filing_text(data["cik"], f)
        except Exception as e:  # keep going on a single bad filing
            log(f"  skipped ({e})")
            continue
        period = f" (period {f['period']})" if f.get("period") else ""
        data["docs"][doc_id] = {
            "id": doc_id, "source": "SEC EDGAR", "doc_type": f["form"],
            "label": FORM_LABELS.get(f["form"], f["form"]) + period, "date": f["date"],
            "url": url, "text": text, "extraction": None, "added": now(),
        }
        new_ids.append(doc_id)
    return new_ids


def extract_pending(data: dict, agent: ConsistencyAgent, doc_ids=None, log=print) -> int:
    """Run statement extraction (oldest first, so topic labels stay stable)."""
    pending = [d for d in data["docs"].values()
               if not d.get("extraction") and (doc_ids is None or d["id"] in doc_ids)]
    for i, doc in enumerate(sorted(pending, key=lambda d: d["date"]), 1):
        log(f"Extracting statements {i}/{len(pending)}: {doc['date']} {doc['label']}")
        doc["extraction"] = agent.extract(data["company"], doc, known_topics(data))
        doc["extracted_at"] = now()
    return len(pending)


def run_consistency(data: dict, agent: ConsistencyAgent, current_id: str | None = None,
                    baseline_ids: list[str] | None = None) -> dict:
    docs = sorted([d for d in data["docs"].values() if d.get("extraction")], key=lambda d: d["date"])
    if len(docs) < 2:
        raise ValueError("Need at least two analysed documents to compare.")
    current = data["docs"][current_id] if current_id else docs[-1]
    prior = [d for d in docs if d["id"] != current["id"] and d["date"] <= current["date"]]
    if baseline_ids is not None:
        prior = [d for d in prior if d["id"] in baseline_ids]
    if not prior:
        raise ValueError("No earlier documents to compare against.")
    review = agent.compare(data["company"], current, prior)
    analysis = {
        "run_at": now(), "model": agent.model,
        "current_id": current["id"], "current_label": f"{current['date']} {current['label']}",
        "baseline_ids": [d["id"] for d in prior], "review": review,
    }
    data["analyses"].append(analysis)
    return analysis


def report_markdown(data: dict, analysis: dict) -> str:
    r = analysis["review"]
    md = [f"# Management consistency report — {data['company']} ({data['ticker']})",
          f"*Current document:* {analysis['current_label']}  ",
          f"*Compared against:* {len(analysis['baseline_ids'])} earlier documents  ",
          f"*Run:* {analysis['run_at']} with {analysis['model']}", "",
          f"## Consistency score: {r['consistency_score']}/100", r.get("score_rationale", ""), "",
          "## Summary", r["summary"], ""]
    if r.get("tone_shift"):
        md += ["## Tone shift", r["tone_shift"], ""]
    if r.get("red_flags"):
        md += ["## Red flags"] + [f"- {x}" for x in r["red_flags"]] + [""]
    md.append("## Findings")
    for f in sorted(r["findings"], key=lambda f: ["high", "medium", "low"].index(f.get("severity", "low"))):
        md += [f"### {f['topic']} — {f['verdict']} ({f.get('severity')})"]
        if f.get("prior_statement"):
            md.append(f"- **Before** ({f.get('prior_source', '')}): {f['prior_statement']}")
        if f.get("current_statement"):
            md.append(f"- **Now** ({f.get('current_source', '')}): {f['current_statement']}")
        md += [f"- {f['explanation']}", ""]
    if r.get("questions_for_management"):
        md += ["## Questions for management"] + [f"- {q}" for q in r["questions_for_management"]]
    md += ["", "_AI-generated analysis of public statements. Verify against the source documents; not investment advice._"]
    return "\n".join(md)


def monitor_ticker(store: Store, edgar: EdgarClient, agent: ConsistencyAgent, ticker: str,
                   limit: int = 8, log=print) -> dict | None:
    """One monitoring pass: fetch new filings, extract, and review the newest one if anything changed."""
    data = store.load(ticker)
    new_ids = sync_sec(data, edgar, limit=limit, log=log)
    extract_pending(data, agent, log=log)
    analysis = None
    if new_ids and sum(1 for d in data["docs"].values() if d.get("extraction")) >= 2:
        log("New statements found — running consistency review…")
        analysis = run_consistency(data, agent)
    store.save(data)
    return analysis
