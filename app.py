"""Management Consistency Monitor — Streamlit app.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import os
from datetime import date

import pandas as pd
import streamlit as st

from agent import MODELS, VERDICTS, ConsistencyAgent
from alerts import (DEFAULT_SETTINGS, alert_log, build_email, load_settings, notify, save_settings,
                    send_email, send_webhook, smtp_password)
from edgar import EdgarClient
from pipeline import (Store, add_manual_doc, extract_pending, monitor_ticker, parse_tickers, portfolio_row,
                      report_markdown, run_consistency, sync_sec)

st.set_page_config(page_title="Management Consistency Monitor", page_icon="🧭", layout="wide")

VERDICT_ICON = {"consistent": "🟢", "delivered": "🟢", "evolved": "🔵", "new_commitment": "⚪",
                "walked_back": "🟠", "missed": "🔴", "dropped": "🟠", "contradiction": "🔴"}
SEV_ORDER = {"high": 0, "medium": 1, "low": 2}

store = Store()
settings = load_settings()


def secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🧭 Setup")
    api_key = st.text_input("Anthropic API key", value=secret("ANTHROPIC_API_KEY"), type="password")
    model = st.selectbox("Claude model", MODELS)
    user_agent = st.text_input("SEC User-Agent (name + email)", value=secret("SEC_USER_AGENT"),
                               help="SEC EDGAR requires a contact, e.g. 'Jane Doe jane@fund.com'.")
    st.divider()
    st.subheader("Companies")
    default_list = ", ".join(store.watchlist() or store.tickers() or ["AAPL", "MSFT"])
    tickers_text = st.text_area("Tickers — one or many", value=default_list, height=80,
                                help="Separate with commas, spaces or new lines, e.g. AAPL, MSFT, NVDA")
    tickers = parse_tickers(tickers_text)
    if tickers != store.watchlist():
        store.set_watchlist(tickers)  # the list is the monitored watchlist
    st.caption(f"Monitoring {len(tickers)} compan{'y' if len(tickers) == 1 else 'ies'}.")
    ticker = st.selectbox("Company to inspect", tickers) if tickers else None
    with st.expander("SEC fetch options"):
        forms = st.multiselect("Filing types", ["8-K", "10-Q", "10-K"], default=["8-K", "10-Q", "10-K"])
        limit = st.slider("Recent filings per company", 2, 30, 8)
        earnings_only = st.checkbox("8-K: earnings releases only (Item 2.02)", value=True)
    st.caption("History is saved in ./data, so every run builds on the last.")

smtp_pwd = st.session_state.get("smtp_pwd", "") or secret("SMTP_PASSWORD")


def need_agent() -> ConsistencyAgent | None:
    if not api_key:
        st.error("Add your Anthropic API key in the sidebar.")
        return None
    return ConsistencyAgent(api_key, model)


def need_edgar() -> EdgarClient | None:
    try:
        return EdgarClient(user_agent)
    except ValueError as e:
        st.error(str(e))
        return None


def alert_after_review(data: dict, analysis: dict, log=st.write) -> None:
    reasons = notify(data, analysis, settings, report_markdown(data, analysis), smtp_pwd, log=log)
    if reasons:
        st.toast(f"🔔 {data['ticker']}: alert sent — {reasons[0]}")


def read_upload(f) -> str:
    name = f.name.lower()
    raw = f.read()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages)
    if name.endswith(".docx"):
        import docx
        return "\n".join(p.text for p in docx.Document(io.BytesIO(raw)).paragraphs)
    if name.endswith((".htm", ".html")):
        from edgar import html_to_text
        return html_to_text(raw.decode("utf-8", "ignore"))
    return raw.decode("utf-8", "ignore")


st.title("🧭 Management Consistency Monitor")
st.caption("Tracks what management of public companies says and checks it against what they said before.")

tab_port, tab_src, tab_stmt, tab_report, tab_timeline, tab_alerts = st.tabs(
    ["🌐 Portfolio", "📥 Sources", "🗒️ Statements", "📊 Consistency report", "🕰️ Topic timeline",
     "🔔 Alerts & settings"])

# ================================================================ portfolio
with tab_port:
    if not tickers:
        st.info("Enter one or more tickers in the sidebar.")
    else:
        rows = [portfolio_row(store.load(t)) for t in tickers]
        pdf = pd.DataFrame(rows)
        st.dataframe(
            pdf, hide_index=True, width="stretch",
            column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                "Change": st.column_config.NumberColumn("Δ vs previous", format="%+d"),
            })
        c1, c2 = st.columns([1, 2])
        run_all = c1.button(f"▶️ Run monitor on all {len(tickers)}", type="primary")
        force = c2.checkbox("Re-review even when there are no new filings", value=False)
        st.caption("For each company: fetch new SEC filings → extract statements → consistency review → "
                   "send an alert if a rule in *Alerts & settings* is triggered.")
        if run_all:
            agent, edgar = need_agent(), need_edgar()
            if agent and edgar:
                prog = st.progress(0.0)
                for n, t in enumerate(tickers, 1):
                    with st.status(t, expanded=False) as status:
                        try:
                            a = monitor_ticker(store, edgar, agent, t, limit, log=st.write)
                            data_t = store.load(t)
                            if a is None and force and sum(1 for d in data_t["docs"].values() if d.get("extraction")) >= 2:
                                a = run_consistency(data_t, agent)
                                store.save(data_t)
                            if a:
                                alert_after_review(data_t, a)
                                status.update(label=f"{t}: score {a['review']['consistency_score']}/100", state="complete")
                            else:
                                status.update(label=f"{t}: no new filings", state="complete")
                        except Exception as e:
                            status.update(label=f"{t}: error — {e}", state="error")
                    prog.progress(n / len(tickers))
                st.rerun()
        scored = pdf.dropna(subset=["Score"])
        if not scored.empty:
            st.markdown("#### Latest consistency scores")
            st.bar_chart(scored.set_index("Ticker")["Score"])

data = store.load(ticker) if ticker else None

# ================================================================ sources
with tab_src:
    if not data:
        st.info("Enter a ticker in the sidebar.")
    else:
        st.markdown(f"**{data['company']}** ({ticker}) · {len(data['docs'])} documents · "
                    f"{sum(1 for d in data['docs'].values() if d.get('extraction'))} analysed")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("From SEC EDGAR")
            st.write("Earnings press releases (8-K Ex. 99) and the MD&A section of 10-Qs/10-Ks.")
            if st.button("Fetch latest filings", type="primary"):
                edgar = need_edgar()
                if edgar:
                    with st.status("Fetching from EDGAR…", expanded=True) as status:
                        try:
                            new = sync_sec(data, edgar, tuple(forms), limit, earnings_only, log=st.write)
                            store.save(data)
                            status.update(label=f"Added {len(new)} new filings", state="complete")
                        except Exception as e:
                            status.update(label=f"EDGAR error: {e}", state="error")
        with c2:
            st.subheader("Add your own")
            st.write("Earnings call transcripts, investor-day decks, interviews, shareholder letters.")
            with st.form("manual", clear_on_submit=True):
                label = st.text_input("Label", placeholder="Q2 2026 earnings call")
                doc_type = st.selectbox("Type", ["Earnings call transcript", "Investor day", "Interview",
                                                 "Shareholder letter", "Conference presentation", "Other"])
                d = st.date_input("Date", value=date.today())
                up = st.file_uploader("File (txt, pdf, docx, html)", type=["txt", "md", "pdf", "docx", "htm", "html"])
                pasted = st.text_area("…or paste text", height=120)
                url = st.text_input("Source URL (optional)")
                if st.form_submit_button("Add document"):
                    text = read_upload(up) if up else pasted
                    if not (label and text.strip()):
                        st.warning("Give it a label and some text.")
                    else:
                        add_manual_doc(data, label, doc_type, d.isoformat(), text, url)
                        store.save(data)
                        st.success(f"Added '{label}'.")

        st.divider()
        if data["docs"]:
            docs = sorted(data["docs"].values(), key=lambda x: x["date"], reverse=True)
            table = pd.DataFrame([{
                "Analyse": not d.get("extraction"), "Date": d["date"], "Document": d["label"],
                "Source": d["source"], "Chars": len(d["text"]),
                "Statements": len(d["extraction"]["statements"]) if d.get("extraction") else None,
                "Tone": d["extraction"].get("tone_score") if d.get("extraction") else None,
                "URL": d.get("url") or None, "id": d["id"]} for d in docs])
            edited = st.data_editor(
                table, hide_index=True, width="stretch",
                disabled=[c for c in table.columns if c != "Analyse"],
                column_config={"id": None, "URL": st.column_config.LinkColumn("URL", display_text="open"),
                               "Analyse": st.column_config.CheckboxColumn("Analyse", help="Extract (or re-extract) statements")})
            selected = edited.loc[edited["Analyse"], "id"].tolist()
            b1, b2 = st.columns([1, 1])
            if b1.button(f"🤖 Extract statements from {len(selected)} selected", disabled=not selected):
                agent = need_agent()
                if agent:
                    for i in selected:
                        data["docs"][i]["extraction"] = None
                    with st.status("Reading documents…", expanded=True) as status:
                        try:
                            n = extract_pending(data, agent, set(selected), log=st.write)
                            status.update(label=f"Extracted statements from {n} documents", state="complete")
                        except Exception as e:
                            status.update(label=f"Error: {e}", state="error")
                        store.save(data)
                    st.rerun()
            with b2.popover("🗑️ Remove a document"):
                rm = st.selectbox("Document", [f"{d['date']} {d['label']}|{d['id']}" for d in docs])
                if st.button("Remove"):
                    data["docs"].pop(rm.split("|")[-1], None)
                    store.save(data)
                    st.rerun()
        else:
            st.info("No documents yet — fetch from EDGAR or add a transcript.")

# ================================================================ statements
with tab_stmt:
    rows = [{"Date": d["date"], "Document": d["label"], **s}
            for d in (data["docs"].values() if data else []) if d.get("extraction")
            for s in d["extraction"].get("statements", [])]
    if not rows:
        st.info("Extract statements on the Sources tab first.")
    else:
        df = pd.DataFrame(rows).sort_values("Date", ascending=False)
        f1, f2, f3 = st.columns(3)
        cats = f1.multiselect("Category", sorted(df["category"].dropna().unique()))
        fwd = f2.selectbox("Statements", ["All", "Forward-looking only", "Backward-looking only"])
        q = f3.text_input("Search")
        if cats:
            df = df[df["category"].isin(cats)]
        if fwd != "All":
            df = df[df["forward_looking"] == (fwd == "Forward-looking only")]
        if q:
            df = df[df.apply(lambda r: q.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
        cols = [c for c in ["Date", "Document", "topic", "category", "statement", "value", "timeframe",
                            "forward_looking", "speaker", "quote"] if c in df.columns]
        st.dataframe(df[cols], hide_index=True, width="stretch", height=520)
        st.download_button("Download CSV", df[cols].to_csv(index=False), f"{ticker}_statements.csv")
        with st.expander("Document summaries & tone"):
            for d in sorted(data["docs"].values(), key=lambda x: x["date"], reverse=True):
                if d.get("extraction"):
                    e = d["extraction"]
                    st.markdown(f"**{d['date']} — {d['label']}** · tone {e.get('tone_score')}  \n{e.get('summary', '')}")

# ================================================================ report
with tab_report:
    analysed = sorted([d for d in data["docs"].values() if d.get("extraction")], key=lambda x: x["date"],
                      reverse=True) if data else []
    if len(analysed) < 2:
        st.info("You need at least two analysed documents to compare.")
    else:
        labels = {d["id"]: f"{d['date']} — {d['label']}" for d in analysed}
        c1, c2 = st.columns([1, 2])
        current_id = c1.selectbox("Statement under review", list(labels), format_func=labels.get)
        cur_date = data["docs"][current_id]["date"]
        earlier = [i for i in labels if i != current_id and data["docs"][i]["date"] <= cur_date]
        baseline = c2.multiselect("Compare against", earlier, default=earlier, format_func=labels.get)
        send_alerts = st.checkbox("Send alerts if this review triggers a rule", value=True)
        if st.button("🔍 Run consistency review", type="primary", disabled=not baseline):
            agent = need_agent()
            if agent:
                with st.spinner("Comparing statements…"):
                    try:
                        a = run_consistency(data, agent, current_id, baseline)
                        store.save(data)
                        if send_alerts:
                            alert_after_review(data, a, log=lambda *_: None)
                    except Exception as e:
                        st.error(f"Error: {e}")

        if data["analyses"]:
            runs = list(reversed(data["analyses"]))
            pick = st.selectbox("Report", range(len(runs)),
                                format_func=lambda i: f"{runs[i]['current_label']}  ·  run {runs[i]['run_at']}")
            a = runs[pick]
            r = a["review"]
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Consistency score", f"{r['consistency_score']}/100")
            vc = pd.Series([f["verdict"] for f in r["findings"]], dtype=object).value_counts()
            m2.metric("Contradictions / misses", int(vc.get("contradiction", 0) + vc.get("missed", 0)))
            m3.metric("Walked back / dropped", int(vc.get("walked_back", 0) + vc.get("dropped", 0)))
            m4.metric("Consistent / delivered", int(vc.get("consistent", 0) + vc.get("delivered", 0)))
            st.progress(r["consistency_score"] / 100)
            if r.get("score_rationale"):
                st.caption(r["score_rationale"])
            st.markdown("#### Summary")
            st.write(r["summary"])
            if r.get("tone_shift"):
                st.markdown(f"**Tone shift:** {r['tone_shift']}")
            if r.get("red_flags"):
                st.markdown("#### 🚩 Red flags")
                for x in r["red_flags"]:
                    st.warning(x)

            if r["findings"]:
                st.markdown("#### Findings")
                fdf = pd.DataFrame(r["findings"])
                fdf["sev"] = fdf["severity"].map(SEV_ORDER)
                fdf = fdf.sort_values(["sev", "verdict"]).drop(columns="sev")
                vsel = st.multiselect("Verdicts", list(VERDICTS), default=sorted(fdf["verdict"].unique()),
                                      format_func=lambda v: f"{VERDICT_ICON[v]} {v}")
                for f in fdf[fdf["verdict"].isin(vsel)].to_dict("records"):
                    with st.expander(f"{VERDICT_ICON.get(f['verdict'], '')} **{f['topic']}** — {f['verdict']} · {f['severity']}"):
                        if f.get("prior_statement"):
                            st.markdown(f"**Before** · _{f.get('prior_source') or ''}_  \n{f['prior_statement']}")
                        if f.get("current_statement"):
                            st.markdown(f"**Now** · _{f.get('current_source') or ''}_  \n{f['current_statement']}")
                        st.info(f["explanation"])
            if r.get("questions_for_management"):
                st.markdown("#### ❓ Questions for management")
                for q in r["questions_for_management"]:
                    st.markdown(f"- {q}")
            d1, d2 = st.columns(2)
            d1.download_button("Download report (Markdown)", report_markdown(data, a),
                               f"{ticker}_consistency_{a['current_label'][:10]}.md")
            if r["findings"]:
                d2.download_button("Download findings (CSV)", pd.DataFrame(r["findings"]).to_csv(index=False),
                                   f"{ticker}_findings.csv")
            with st.expander("Verdict legend"):
                for k, v in VERDICTS.items():
                    st.markdown(f"{VERDICT_ICON[k]} **{k}** — {v}")

        if len(data["analyses"]) > 1:
            st.markdown("#### Score history")
            hist = pd.DataFrame([{"Document": x["current_label"], "Score": x["review"]["consistency_score"]}
                                 for x in data["analyses"]]).drop_duplicates("Document", keep="last")
            st.line_chart(hist.set_index("Document"))
        st.caption("AI-generated analysis of public statements. Verify against source documents; not investment advice.")

# ================================================================ timeline
with tab_timeline:
    rows = [{"Date": d["date"], "Document": d["label"], **s}
            for d in (data["docs"].values() if data else []) if d.get("extraction")
            for s in d["extraction"].get("statements", [])]
    if not rows:
        st.info("Extract statements first.")
    else:
        df = pd.DataFrame(rows)
        counts = df.groupby("topic")["Date"].nunique().sort_values(ascending=False)
        topic = st.selectbox("Topic (most-discussed first)", counts.index,
                             format_func=lambda t: f"{t}  ({counts[t]} documents)")
        for s in df[df["topic"] == topic].sort_values("Date").to_dict("records"):
            fwd = "🔮 forward-looking" if s.get("forward_looking") else "📌 statement"
            meta = " · ".join(str(s[k]) for k in ("value", "timeframe", "speaker") if isinstance(s.get(k), str) and s.get(k))
            st.markdown(f"**{s['Date']}** — {s['Document']}  ·  {fwd}  \n{s['statement']}"
                        + (f"  \n`{meta}`" if meta else "")
                        + (f"  \n> {s['quote']}" if isinstance(s.get("quote"), str) and s.get("quote") else ""))
        tones = pd.DataFrame([{"Date": d["date"], "Tone": d["extraction"].get("tone_score")}
                              for d in data["docs"].values() if d.get("extraction")]).sort_values("Date")
        st.markdown("#### Management tone over time (-5 … +5)")
        st.line_chart(tones.set_index("Date"))

# ================================================================ alerts & settings
with tab_alerts:
    st.subheader("🔔 Automatic alerts")
    st.write("After every consistency review (from the Portfolio run, a manual review, or the scheduled "
             "`monitor.py`), the agent checks these rules and alerts you if any is triggered.")
    with st.form("alert_settings"):
        enabled = st.toggle("Alerts enabled", value=settings["enabled"])
        st.markdown("**When to alert** — any one of these triggers an alert")
        r1, r2 = st.columns(2)
        min_score = r1.slider("Score falls below", 0, 100, int(settings["min_score"]))
        max_drop = r2.slider("Score drops by at least (points vs previous review)", 1, 50, int(settings["max_drop"]))
        alert_verdicts = st.multiselect("Findings that trigger an alert", list(VERDICTS),
                                        default=settings["alert_verdicts"],
                                        format_func=lambda v: f"{VERDICT_ICON[v]} {v.replace('_', ' ')}")
        r3, r4, r5 = st.columns(3)
        high_only = r3.checkbox("Only high-severity findings", value=settings["high_severity_only"])
        on_flags = r4.checkbox("Any red flag", value=settings["alert_on_red_flags"])
        digest = r5.checkbox("Send a summary after every review", value=settings["send_digest_without_trigger"],
                             help="Email even when nothing is triggered.")

        st.markdown("**Email**")
        recipients = st.text_input("Send alerts to (comma-separated)", value=", ".join(settings["recipients"]))
        e1, e2, e3 = st.columns([2, 1, 1])
        smtp_host = e1.text_input("SMTP server", value=settings["smtp_host"])
        smtp_port = e2.number_input("Port", value=int(settings["smtp_port"]), step=1)
        smtp_security = e3.selectbox("Security", ["STARTTLS", "SSL", "NONE"],
                                     index=["STARTTLS", "SSL", "NONE"].index(settings["smtp_security"]))
        e4, e5 = st.columns(2)
        smtp_user = e4.text_input("SMTP username", value=settings["smtp_user"])
        sender = e5.text_input("From address (blank = username)", value=settings["sender"])
        pwd_in = st.text_input("SMTP password / app password", type="password",
                               help="Not saved to disk. Set SMTP_PASSWORD in your environment or "
                                    ".streamlit/secrets.toml so scheduled runs can send email too.")
        st.markdown("**Chat (optional)**")
        webhook = st.text_input("Slack or Discord incoming-webhook URL", value=settings["webhook_url"])
        if st.form_submit_button("💾 Save settings", type="primary"):
            settings.update({
                "enabled": enabled, "min_score": min_score, "max_drop": max_drop,
                "alert_verdicts": alert_verdicts, "high_severity_only": high_only,
                "alert_on_red_flags": on_flags, "send_digest_without_trigger": digest,
                "recipients": [x.strip() for x in recipients.replace(";", ",").split(",") if x.strip()],
                "smtp_host": smtp_host.strip(), "smtp_port": int(smtp_port), "smtp_security": smtp_security,
                "smtp_user": smtp_user.strip(), "sender": sender.strip(), "webhook_url": webhook.strip(),
            })
            save_settings(settings)
            if pwd_in:
                st.session_state.smtp_pwd = pwd_in
            st.success("Settings saved.")
    if not smtp_password(smtp_pwd) and settings["smtp_user"]:
        st.warning("No SMTP password set — enter it above (this session) or set SMTP_PASSWORD.")

    t1, t2 = st.columns(2)
    if t1.button("✉️ Send test email", disabled=not settings["recipients"]):
        try:
            send_email(settings, "Management Consistency Monitor — test alert",
                       "<p>✅ Alerts are working. You'll receive messages like this when management's "
                       "consistency changes.</p>", "Alerts are working.", password=smtp_pwd)
            st.success(f"Test email sent to {', '.join(settings['recipients'])}.")
        except Exception as e:
            st.error(f"Email failed: {e}")
    if t2.button("💬 Send test webhook", disabled=not settings["webhook_url"]):
        try:
            send_webhook(settings["webhook_url"], "✅ Management Consistency Monitor: test alert")
            st.success("Webhook sent.")
        except Exception as e:
            st.error(f"Webhook failed: {e}")
    if data and data["analyses"]:
        with st.expander(f"Preview the alert for {ticker}'s latest review"):
            from alerts import evaluate
            a = data["analyses"][-1]
            reasons = evaluate(data, a, settings)
            st.write("Would trigger:" if reasons else "Would **not** trigger with current rules.")
            for x in reasons:
                st.markdown(f"- {x}")
            st.html(build_email(data, a, reasons, "")[1])

    st.subheader("Alert history")
    log = alert_log()
    if log:
        st.dataframe(pd.DataFrame(reversed(log)), hide_index=True, width="stretch")
    else:
        st.caption("No alerts sent yet.")

    with st.expander("ℹ️ Running the monitor automatically"):
        st.markdown("""
The app only runs while it's open. To monitor continuously, schedule **`monitor.py`** — it checks every
ticker in your list, reviews new filings and sends alerts with the settings saved here:

```bash
export ANTHROPIC_API_KEY=...  SEC_USER_AGENT="Jane Doe jane@fund.com"  SMTP_PASSWORD=...
python monitor.py
```
Cron, weekdays at 7:00 and 18:00 (earnings are usually released before the open or after the close):
`0 7,18 * * 1-5 cd /path/to/mgmt_monitor && python monitor.py`
""")
