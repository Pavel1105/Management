"""Minimal SEC EDGAR client: ticker lookup, filing lists, and clean text extraction."""
from __future__ import annotations

import re
import time

import requests
from bs4 import BeautifulSoup

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}"


class EdgarClient:
    """SEC requires a descriptive User-Agent (name + email) and <=10 requests/sec."""

    def __init__(self, user_agent: str):
        if not user_agent or "@" not in user_agent:
            raise ValueError("SEC requires a User-Agent like 'Jane Doe jane@example.com'.")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self._last_call = 0.0
        self._tickers: dict | None = None

    def _get(self, url: str) -> requests.Response:
        wait = 0.15 - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        resp = self.session.get(url, timeout=30)
        self._last_call = time.time()
        resp.raise_for_status()
        return resp

    def lookup(self, ticker: str) -> tuple[str, str]:
        """Return (10-digit CIK, company name) for a ticker."""
        if self._tickers is None:
            self._tickers = self._get(TICKERS_URL).json()
        t = ticker.strip().upper()
        for row in self._tickers.values():
            if row["ticker"].upper() == t:
                return str(row["cik_str"]).zfill(10), row["title"]
        raise ValueError(f"Ticker '{ticker}' not found on SEC EDGAR.")

    def list_filings(self, cik: str, forms=("8-K", "10-Q", "10-K"), limit: int = 8,
                     earnings_only: bool = True) -> list[dict]:
        """Most recent filings of the given forms. 8-Ks are limited to Item 2.02
        (results of operations = earnings releases) when earnings_only is True."""
        recent = self._get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
        n = len(recent["form"])
        items_col = recent.get("items") or [""] * n
        out = []
        for i in range(n):
            form = recent["form"][i]
            if form not in forms:
                continue
            items = items_col[i] or ""
            if form == "8-K" and earnings_only and "2.02" not in items:
                continue
            out.append({
                "form": form,
                "date": recent["filingDate"][i],
                "period": recent["reportDate"][i],
                "accession": recent["accessionNumber"][i],
                "primary": recent["primaryDocument"][i],
                "items": items,
            })
            if len(out) >= limit:
                break
        return out

    def filing_text(self, cik: str, filing: dict) -> tuple[str, str]:
        """Return (text, url). For 8-Ks: the Exhibit 99 press release / commentary.
        For 10-K/10-Q: the MD&A section (falls back to the full document)."""
        acc = filing["accession"].replace("-", "")
        base = ARCHIVE_URL.format(cik=int(cik), acc=acc)
        docs = [filing["primary"]]
        if filing["form"] == "8-K":
            index = self._get(f"{base}/index.json").json()
            names = [it["name"] for it in index["directory"]["item"]]
            exhibits = [n for n in names
                        if re.search(r"ex[-_]?99", n, re.I) and n.lower().endswith((".htm", ".html", ".txt"))]
            if exhibits:
                docs = sorted(exhibits)[:3]
        text = "\n\n".join(html_to_text(self._get(f"{base}/{d}").text) for d in docs)
        if filing["form"] in ("10-K", "10-Q"):
            text = extract_mdna(text)
        return text, f"{base}/{docs[0]}"


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    for tag in soup.find_all(["ix:header"]):  # inline-XBRL hidden metadata
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_mdna(text: str) -> str:
    """Pick the longest 'Management's Discussion and Analysis' span (skips the table of contents)."""
    starts = [m.start() for m in re.finditer(
        r"item\s*[27]\s*[.:\-]?\s*management.{0,3}s\s+discussion\s+and\s+analysis", text, re.I)]
    best = ""
    for s in starts:
        end_m = re.search(r"item\s*(3|7a)\s*[.:\-]?\s*quantitative\s+and\s+qualitative", text[s + 200:], re.I)
        end = s + 200 + end_m.start() if end_m else min(len(text), s + 200_000)
        if end - s > len(best):
            best = text[s:end]
    return best if len(best) > 2000 else text
