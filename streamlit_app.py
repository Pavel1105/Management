"""The AI agent: extracts management statements and judges consistency over time (Claude API)."""
from __future__ import annotations

import json

import anthropic

MODELS = ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"]
MAX_DOC_CHARS = 180_000

CATEGORIES = ["guidance", "financial_target", "reported_result", "strategy", "capital_allocation",
              "product_timeline", "market_outlook", "operations", "risk", "explanation"]

VERDICTS = {
    "consistent": "Repeats or aligns with prior statements",
    "delivered": "A prior forward-looking claim was met",
    "evolved": "Changed, but the change is acknowledged and explained",
    "new_commitment": "A material new claim with no prior counterpart",
    "walked_back": "Softened or narrowed a prior claim without clear acknowledgement",
    "missed": "A prior forward-looking claim was not met",
    "dropped": "A prior priority/commitment is no longer mentioned",
    "contradiction": "Directly conflicts with a prior statement",
}

EXTRACT_TOOL = {
    "name": "record_statements",
    "description": "Record the material statements management made in this document.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "3-4 sentence summary of management's message."},
            "tone_score": {"type": "integer", "minimum": -5, "maximum": 5,
                           "description": "-5 very defensive/pessimistic ... +5 very confident/optimistic"},
            "tone_note": {"type": "string"},
            "statements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Short canonical label, e.g. 'FY revenue guidance', 'Gross margin', 'Share buybacks'."},
                        "category": {"type": "string", "enum": CATEGORIES},
                        "statement": {"type": "string", "description": "Concise paraphrase of what management said."},
                        "quote": {"type": "string", "description": "Short verbatim excerpt (max ~25 words) as evidence."},
                        "metric": {"type": "string"},
                        "value": {"type": "string", "description": "Number/range/target if any, with units."},
                        "timeframe": {"type": "string", "description": "e.g. 'Q4 2026', 'FY2027', 'by 2028'."},
                        "forward_looking": {"type": "boolean"},
                        "speaker": {"type": "string", "description": "CEO/CFO/name if known, else 'Company'."},
                    },
                    "required": ["topic", "category", "statement", "forward_looking"],
                },
            },
        },
        "required": ["summary", "tone_score", "statements"],
    },
}

COMPARE_TOOL = {
    "name": "record_consistency_review",
    "description": "Record the consistency review of the current document against prior statements.",
    "input_schema": {
        "type": "object",
        "properties": {
            "consistency_score": {"type": "integer", "minimum": 0, "maximum": 100},
            "score_rationale": {"type": "string"},
            "summary": {"type": "string", "description": "Analyst-style paragraph on how consistent management has been."},
            "tone_shift": {"type": "string"},
            "red_flags": {"type": "array", "items": {"type": "string"}},
            "questions_for_management": {"type": "array", "items": {"type": "string"}},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "verdict": {"type": "string", "enum": list(VERDICTS)},
                        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                        "prior_statement": {"type": "string"},
                        "prior_source": {"type": "string"},
                        "current_statement": {"type": "string"},
                        "current_source": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["topic", "verdict", "severity", "explanation"],
                },
            },
        },
        "required": ["consistency_score", "summary", "findings"],
    },
}

EXTRACT_SYSTEM = """You are a meticulous buy-side equity analyst. You read company communications and
record every material claim management makes, so that it can later be checked against what they say
in future. Capture: numeric guidance (with ranges and timeframes), financial/operating targets,
reported results that relate to earlier guidance, strategic priorities, capital allocation (buybacks,
dividends, M&A, capex, debt), product and project timelines, demand/market outlook, how risks are
characterised, and management's explanations for performance. Ignore boilerplate safe-harbor language
and pure accounting definitions. Be faithful: never invent numbers."""

COMPARE_SYSTEM = """You are a skeptical buy-side analyst assessing whether a public company's management
is consistent over time. Compare the CURRENT document's statements with the PRIOR statements.

Verdicts:
""" + "\n".join(f"- {k}: {v}" for k, v in VERDICTS.items()) + """

Rules:
- Match statements by subject, not by exact wording; topic labels may differ slightly.
- Check prior forward-looking claims whose timeframe has now passed against reported results (delivered/missed).
- Distinguish transparent, explained revisions (evolved) from silent shifts (walked_back/dropped).
- Only flag 'dropped' for priorities that were emphasised previously and are clearly relevant now
  (note that a press release may legitimately omit topics an MD&A covers).
- Severity reflects investor relevance: guidance, margins, capital allocation and strategy are high.
- Score 0-100: 90+ highly consistent and credible; 70-89 mostly consistent with explained changes;
  50-69 notable unexplained shifts or misses; <50 repeated contradictions or credibility concerns.
- Cite sources using the labels given. Be specific and evidence-based; do not speculate beyond the text."""


class ConsistencyAgent:
    def __init__(self, api_key: str, model: str = MODELS[0]):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _call_tool(self, system: str, prompt: str, tool: dict, max_tokens: int) -> dict:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in resp.content:
            if block.type == "tool_use":
                return block.input
        raise RuntimeError("Model did not return structured output.")

    def extract(self, company: str, doc: dict, known_topics: list[str]) -> dict:
        text = doc["text"][:MAX_DOC_CHARS]
        topics = ", ".join(sorted(set(known_topics))[:150]) or "(none yet)"
        prompt = f"""Company: {company}
Document: {doc['label']} ({doc['doc_type']}), dated {doc['date']}

Topic labels already used for this company (reuse them when the subject is the same): {topics}

<document>
{text}
</document>

Extract up to 60 of the most material management statements using the record_statements tool."""
        return self._call_tool(EXTRACT_SYSTEM, prompt, EXTRACT_TOOL, max_tokens=12_000)

    def compare(self, company: str, current: dict, prior_docs: list[dict]) -> dict:
        prompt = f"""Company: {company}

CURRENT DOCUMENT — {source_label(current)}
Summary: {current['extraction'].get('summary', '')}
Tone: {current['extraction'].get('tone_score')} ({current['extraction'].get('tone_note', '')})
Statements:
{format_statements(current)}

PRIOR STATEMENTS (oldest first):
"""
        for d in sorted(prior_docs, key=lambda x: x["date"]):
            prompt += f"\n--- {source_label(d)} | tone {d['extraction'].get('tone_score')} ---\n"
            prompt += format_statements(d) + "\n"
        prompt += "\nProduce the consistency review with the record_consistency_review tool."
        return self._call_tool(COMPARE_SYSTEM, prompt, COMPARE_TOOL, max_tokens=16_000)


def source_label(doc: dict) -> str:
    return f"{doc['date']} {doc['label']}"


def format_statements(doc: dict) -> str:
    lines = []
    for s in doc["extraction"].get("statements", []):
        extra = "; ".join(f"{k}={s[k]}" for k in ("value", "timeframe", "speaker") if s.get(k))
        fl = "FWD" if s.get("forward_looking") else "   "
        lines.append(f"[{fl}] ({s.get('category')}) {s.get('topic')}: {s.get('statement')}"
                     + (f" [{extra}]" if extra else ""))
    return "\n".join(lines) or "(no statements)"


def to_json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
