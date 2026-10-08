"""Alert settings, rules and delivery (email via SMTP, optional Slack/Discord webhook)."""
from __future__ import annotations

import html
import json
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

import requests

from pipeline import DATA_DIR, now

SETTINGS_PATH = DATA_DIR / "settings.json"
ALERT_LOG_PATH = DATA_DIR / "alerts_log.json"

DEFAULT_SETTINGS = {
    "enabled": True,
    "recipients": [],                 # list of email addresses
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_security": "STARTTLS",      # STARTTLS | SSL | NONE
    "smtp_user": "",
    "sender": "",
    "webhook_url": "",                # Slack / Discord incoming webhook (optional)
    # --- rules: an alert fires when ANY of these is true
    "min_score": 60,                  # score below this
    "max_drop": 10,                   # score fell by at least this vs previous review
    "alert_verdicts": ["contradiction", "missed", "walked_back", "dropped"],
    "high_severity_only": True,       # count only high-severity findings for the verdict rule
    "alert_on_red_flags": False,      # any red flag at all
    "send_digest_without_trigger": False,  # also email a short note after every review
}


# ---------------------------------------------------------------- settings
def load_settings(path: Path = SETTINGS_PATH) -> dict:
    s = dict(DEFAULT_SETTINGS)
    if path.exists():
        s.update(json.loads(path.read_text()))
    return s


def save_settings(settings: dict, path: Path = SETTINGS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in settings.items() if k in DEFAULT_SETTINGS}  # never persist passwords
    path.write_text(json.dumps(clean, indent=2))


def smtp_password(override: str = "") -> str:
    return override or os.getenv("SMTP_PASSWORD", "")


# ---------------------------------------------------------------- rules
def previous_analysis(data: dict, analysis: dict) -> dict | None:
    runs = data.get("analyses", [])
    earlier = [a for a in runs if a is not analysis and a["run_at"] <= analysis["run_at"]]
    return earlier[-1] if earlier else None


def evaluate(data: dict, analysis: dict, settings: dict) -> list[str]:
    """Return human-readable reasons why this review should trigger an alert (empty = no alert)."""
    r = analysis["review"]
    score = r["consistency_score"]
    reasons = []
    prev = previous_analysis(data, analysis)
    if score < settings["min_score"]:
        reasons.append(f"Consistency score {score}/100 is below your threshold of {settings['min_score']}.")
    if prev:
        drop = prev["review"]["consistency_score"] - score
        if drop >= settings["max_drop"]:
            reasons.append(f"Score fell {drop} points (from {prev['review']['consistency_score']} to {score}).")
    hits = [f for f in r.get("findings", [])
            if f["verdict"] in settings["alert_verdicts"]
            and (not settings["high_severity_only"] or f.get("severity") == "high")]
    if hits:
        topics = ", ".join(f"{f['topic']} ({f['verdict'].replace('_', ' ')})" for f in hits[:6])
        reasons.append(f"{len(hits)} flagged finding(s): {topics}.")
    if settings["alert_on_red_flags"] and r.get("red_flags"):
        reasons.append(f"{len(r['red_flags'])} red flag(s) raised.")
    return reasons


# ---------------------------------------------------------------- delivery
def build_email(data: dict, analysis: dict, reasons: list[str], report_md: str) -> tuple[str, str, str]:
    r = analysis["review"]
    t = data["ticker"]
    prev = previous_analysis(data, analysis)
    delta = ""
    if prev:
        d = r["consistency_score"] - prev["review"]["consistency_score"]
        delta = f" ({'+' if d >= 0 else ''}{d})"
    subject = (f"⚠️ {t}: management consistency alert — score {r['consistency_score']}/100{delta}"
               if reasons else f"{t}: consistency review — score {r['consistency_score']}/100{delta}")
    findings = sorted(r.get("findings", []), key=lambda f: ["high", "medium", "low"].index(f.get("severity", "low")))
    flagged = [f for f in findings if f["verdict"] not in ("consistent", "delivered")][:8]
    esc = html.escape
    rows = "".join(
        f"<tr><td style='padding:6px;border-bottom:1px solid #eee'><b>{esc(f['topic'])}</b><br>"
        f"<span style='color:#666'>{esc(f['verdict'].replace('_', ' '))} · {esc(f.get('severity', ''))}</span></td>"
        f"<td style='padding:6px;border-bottom:1px solid #eee'>{esc(f['explanation'])}</td></tr>" for f in flagged)
    body_html = f"""<div style="font-family:Arial,sans-serif;max-width:720px">
<h2 style="margin-bottom:4px">{esc(data['company'])} ({t})</h2>
<div style="color:#666">Reviewed: {esc(analysis['current_label'])} · {esc(analysis['run_at'])}</div>
<h1 style="margin:12px 0">{r['consistency_score']}/100{esc(delta)}</h1>
{"<h3>Why you're getting this</h3><ul>" + "".join(f"<li>{esc(x)}</li>" for x in reasons) + "</ul>" if reasons else ""}
<h3>Summary</h3><p>{esc(r['summary'])}</p>
{"<h3>Red flags</h3><ul>" + "".join(f"<li>{esc(x)}</li>" for x in r.get('red_flags', [])) + "</ul>" if r.get('red_flags') else ""}
{"<h3>Key findings</h3><table style='border-collapse:collapse;width:100%'>" + rows + "</table>" if rows else ""}
<p style="color:#999;font-size:12px">Full report attached. AI-generated analysis of public statements; verify against source documents. Not investment advice.</p>
</div>"""
    text = f"{subject}\n\n" + ("\n".join(f"- {x}" for x in reasons) + "\n\n" if reasons else "") + r["summary"]
    return subject, body_html, text


def send_email(settings: dict, subject: str, body_html: str, body_text: str,
               attachment: tuple[str, str] | None = None, password: str = "") -> None:
    if not settings["recipients"]:
        raise ValueError("No alert recipients configured.")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings["sender"] or settings["smtp_user"]
    msg["To"] = ", ".join(settings["recipients"])
    msg.set_content(body_text)
    msg.add_alternative(body_html, subtype="html")
    if attachment:
        msg.add_attachment(attachment[1].encode(), maintype="text", subtype="markdown", filename=attachment[0])
    host, port, sec = settings["smtp_host"], int(settings["smtp_port"]), settings["smtp_security"]
    pwd = smtp_password(password)
    ctx = ssl.create_default_context()
    if sec == "SSL":
        server = smtplib.SMTP_SSL(host, port, context=ctx, timeout=30)
    else:
        server = smtplib.SMTP(host, port, timeout=30)
        if sec == "STARTTLS":
            server.starttls(context=ctx)
    with server:
        if settings["smtp_user"]:
            server.login(settings["smtp_user"], pwd)
        server.send_message(msg)


def send_webhook(url: str, text: str) -> None:
    # "text" is read by Slack, "content" by Discord.
    requests.post(url, json={"text": text, "content": text[:1900]}, timeout=15).raise_for_status()


def log_alert(entry: dict) -> None:
    log = json.loads(ALERT_LOG_PATH.read_text()) if ALERT_LOG_PATH.exists() else []
    log.append(entry)
    ALERT_LOG_PATH.write_text(json.dumps(log[-500:], indent=1))


def alert_log() -> list[dict]:
    return json.loads(ALERT_LOG_PATH.read_text()) if ALERT_LOG_PATH.exists() else []


def notify(data: dict, analysis: dict, settings: dict, report_md: str, password: str = "", log=print) -> list[str]:
    """Evaluate rules and deliver alerts. Returns the reasons (empty list = nothing triggered)."""
    if not settings.get("enabled"):
        return []
    reasons = evaluate(data, analysis, settings)
    if not reasons and not settings.get("send_digest_without_trigger"):
        log(f"{data['ticker']}: no alert (no rule triggered).")
        return []
    subject, body_html, body_text = build_email(data, analysis, reasons, report_md)
    channels, errors = [], []
    if settings["recipients"]:
        try:
            send_email(settings, subject, body_html, body_text,
                       (f"{data['ticker']}_consistency_report.md", report_md), password)
            channels.append("email")
        except Exception as e:
            errors.append(f"email: {e}")
    if settings.get("webhook_url"):
        try:
            send_webhook(settings["webhook_url"], body_text[:3500])
            channels.append("webhook")
        except Exception as e:
            errors.append(f"webhook: {e}")
    log(f"{data['ticker']}: alert sent via {', '.join(channels) or 'nothing'}"
        + (f" — errors: {'; '.join(errors)}" if errors else ""))
    log_alert({"at": now(), "ticker": data["ticker"], "score": analysis["review"]["consistency_score"],
               "document": analysis["current_label"], "triggered": bool(reasons), "reasons": reasons,
               "channels": channels, "errors": errors})
    return reasons
