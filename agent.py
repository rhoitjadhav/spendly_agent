#!/usr/bin/env python3
"""A read-only SQL data-explorer agent on top of a local Ollama model.

Same loop as ../ollama-agent/agent.py, but the tools are swapped for database
exploration, so the model can answer questions about data it has never seen:

    list_tables()      -> table names
    describe_table(t)  -> columns, keys, 3 sample rows
    run_sql(sql)       -> guarded read-only query (see db.py)

Only the standard library is used. Point db.py's get_connection() at your real
database to use this in production.

Usage
-----
    python3 make_demo_db.py                                  # once, to build spendly.db
    python3 agent.py "How much did I spend on Food & Dining in October?"
    python3 agent.py "Total spend per category" --save   # colourful report in reports/*.md
    python3 agent.py --model llama3.2:latest "Total spend per source?"   # faster but weaker
    python3 agent.py --list-tools
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List

import db

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "gemma4:latest"  # most reliable for multi-join SQL; llama3.2:latest is faster but weaker
MAX_STEPS = 14  # explore + query + chart + audit pass all need room

AUDIT_PROMPT = (
    "Audit your draft answer above. For every derived figure in it (sums, shares, percentages, "
    "differences), verify the exact value against the run_sql results in this conversation; if you "
    "cannot confirm a figure exactly, run one more run_sql query to recompute it. Then output the "
    "corrected final answer in the same analyst-briefing format (direct answer first, then data and "
    "charts, then bold Insights bullets). If every figure is already exact, output the answer unchanged."
)

SYSTEM_PROMPT = """You are a careful data analyst answering questions about a personal expense tracker database.
The single table `transactions` stores payments with columns: id, source, date, description, merchant,
amount, category, account, note, raw_text, fingerprint, created_at, updated_at.
Facts about the data:
- `amount` is ALWAYS positive; it is the amount paid, in Indian Rupees (INR).
- `source` is one of: 'navi', 'phonepe', 'googlepay'.
- `date` is the transaction date (ISO format); `created_at`/`updated_at` are timestamps.
You have four tools: list_tables, describe_table, run_sql, make_chart.
Rules:
1. Call list_tables first, then describe_table for the tables you plan to use, before your first query.
2. Only read data (SELECT / WITH). Write standard SQL for SQLite (single quotes for string literals, exact column names from describe_table).
3. Aggregate in SQL (COUNT, SUM, AVG, GROUP BY) instead of pulling many rows; use WHERE to narrow (e.g. by month: strftime('%Y-%m', date)).
4. If run_sql returns an error, read the error, fix the query, and try again.
5. NEVER invent or guess a number. Every figure, table row and chart value must have come from a run_sql result in this conversation. If you cannot compute something, say so.
6. make_chart values must be copied exactly from run_sql results - never re-type or round them differently.
7. Compute derived figures (totals, shares, percentages, differences) with SQL, not by mental arithmetic.

Charts - pick chart_type to match what the data shows (use 1-2 charts, only when they genuinely help):
- comparison or ranking of categories/merchants -> "bar"
- trend over time (months, weeks, days) -> "line"
- part-to-whole composition (share of total) -> "stacked"
- distribution of amounts (counts per range) -> "histogram"
- relationship between two numeric series -> "scatter"
IMPORTANT: call make_chart as an actual tool call. NEVER write chart definitions
or JSON payloads inside your answer text - only paste the rendered ```text blocks
the tool returns (its shares and scales are computed exactly).

Write the final answer like a professional analyst briefing, in Markdown, in this order:

1. DIRECT ANSWER: open with 1-3 plain sentences containing the key figures in ₹. The reply must begin with this paragraph - no heading before it.
2. DATA: markdown table(s) of the supporting rows (proper table syntax with | separators, never raw JSON), then 1-2 charts when comparing/trending (paste make_chart output verbatim). A short ### label above a chart is fine; no other headings.
3. INSIGHTS: a bold **Insights** line, then 2-4 short bullets with patterns, outliers, trends or shares (e.g. "X is 43% of total spend"). Only insights supported by the query results; drop this section entirely if there are none.

Never use numbered headings such as "## 1. Answer".

Emojis: make the answer scannable with a few, used consistently:
- start every Insights bullet with 💡
- use 📈 or 📉 immediately after trend figures in the answer or insights
- use ⚠️ before caveats
- at most one more anywhere (e.g. 📊 before a chart label)
Never put emojis inside tables or chart code blocks. Never exceed ~4 emojis in total.
"""


def build_system_prompt(today: date | None = None) -> str:
    """Anchor 'now' into the prompt. Without this the model cannot reliably
    resolve relative phrases like 'last month' - so Python resolves them
    once, here, and the model never does calendar arithmetic itself."""
    today = today or date.today()
    first_of_month = today.replace(day=1)
    last_month_end = first_of_month - timedelta(days=1)
    return (
        f"{SYSTEM_PROMPT}\n"
        "Context (resolved for you, do not recompute):\n"
        f"- Today is {today.isoformat()} ({today.strftime('%A')}).\n"
        f"- \"This month\" = {today.strftime('%Y-%m')} and \"last month\" = "
        f"{last_month_end.strftime('%Y-%m')} when filtering with strftime('%Y-%m', date).\n"
        f"- The current month is only partially recorded (data ends today); "
        f"for trend comparisons either prefer complete months or note the caveat.\n"
        f"- To check the data's actual range, query it: "
        f"SELECT MIN(date), MAX(date) FROM transactions."
    )


# ---------------------------------------------------------------------------
# Tools (thin wrappers around db.py)
# ---------------------------------------------------------------------------


def list_tables() -> Dict[str, Any]:
    return {"tables": db.list_tables()}


def describe_table(table: str) -> Dict[str, Any]:
    return db.describe_table(table)


def run_sql(sql: str) -> Dict[str, Any]:
    return db.run_sql(sql)


CHART_CHARS = "█▓▒░▄▀●■▲◆"  # distinct fills for stacked segments


def _fmt_amount(v: float, unit: str = "₹") -> str:
    """Compact human format for chart axes: ₹3.85L / ₹106.2k / ₹649."""
    v = float(v)
    if abs(v) >= 1e5:
        return f"{unit}{v / 1e5:,.2f}L"
    if abs(v) >= 1e3:
        return f"{unit}{v / 1e3:,.1f}k"
    if v.is_integer():
        return f"{unit}{v:,.0f}"
    return f"{unit}{v:,.2f}"


def _render_bar(title: str, labels: List[str], values: List[float], unit: str) -> str:
    """Comparison / ranking: horizontal bars, to scale."""
    width = 36
    peak = max(abs(float(v)) for v in values) or 1
    decimals = 0 if all(float(v).is_integer() for v in values) else 2
    lines = [f"{title} ({unit})"]
    for label, value in zip(labels, values):
        v = float(value)
        bar = "█" * (max(1, round(abs(v) / peak * width)) if v != 0 else 0)
        lines.append(f"{label[:20]:<20}│{bar} {unit}{v:,.{decimals}f}")
    return "\n".join(lines)


def _render_line(title: str, labels: List[str], values: List[float], unit: str) -> str:
    """Trend over time: one ● per point on a value grid."""
    vals = [float(v) for v in values]
    if len(vals) < 2:
        raise ValueError("line chart needs at least 2 points")
    rows_n = 8
    lo, hi = min(vals), max(vals)
    if hi == lo:
        hi = lo + 1.0
    cellw = max(2, min(8, max(len(str(l)) for l in labels)))
    grid = [[" "] * len(vals) for _ in range(rows_n)]
    for i, v in enumerate(vals):
        r = round((v - lo) / (hi - lo) * (rows_n - 1))
        grid[rows_n - 1 - r][i] = "●"
    yw = len(_fmt_amount(hi, unit))
    lines = [f"{title} ({unit})"]
    for r in range(rows_n):
        level = lo + (hi - lo) * (rows_n - 1 - r) / (rows_n - 1)
        cells = "".join(c.ljust(cellw) for c in grid[r])
        lines.append(f"{_fmt_amount(level, unit):>{yw}} ┤ {cells}")
    lines.append(" " * (yw + 1) + "└" + "─" * (len(vals) * (cellw + 1) - 1))
    lines.append(" " * (yw + 2) + " ".join(str(l)[:cellw].ljust(cellw) for l in labels))
    return "\n".join(lines)


def _render_stacked(title: str, labels: List[str], values: List[float], unit: str) -> str:
    """Part-to-whole: one segmented 100% bar + legend with code-computed shares."""
    vals = [abs(float(v)) for v in values]
    total = sum(vals)
    if total <= 0:
        raise ValueError("stacked chart needs a positive total")
    width = 40
    counts = [int(v / total * width) for v in vals]
    i = 0
    while sum(counts) < width:  # distribute rounding leftovers
        counts[i % len(counts)] += 1
        i += 1
    bar = "".join(CHART_CHARS[i % len(CHART_CHARS)] * c for i, c in enumerate(counts))
    lines = [f"{title} — composition of {unit}{total:,.2f}", f"[{bar}]"]
    for i in sorted(range(len(vals)), key=lambda j: -vals[j]):
        lines.append(f"{CHART_CHARS[i % len(CHART_CHARS)]} {str(labels[i])[:24]}: "
                     f"{vals[i] / total * 100:.1f}%  ({unit}{vals[i]:,.2f})")
    return "\n".join(lines)


def _render_histogram(title: str, labels: List[str], values: List[float], unit: str) -> str:
    """Distribution: vertical bars, counts per bin."""
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("histogram needs values")
    rows_n = 8
    peak = max(vals) or 1.0
    cellw = max(4, min(12, max(len(str(l)) for l in labels)))
    heights = [max(1, round(v / peak * rows_n)) if v > 0 else 0 for v in vals]
    lines = [f"{title} (count per bin)"]
    for r in range(rows_n, 0, -1):
        cells = "".join(("█" if h >= r else " ").ljust(cellw) for h in heights)
        lines.append(f"{round(peak * r / rows_n):>8} │ {cells}")
    content_w = len(vals) * (cellw + 1) - 1
    lines.append(f"{'':>8} └{'─' * (content_w + 2)}")
    lines.append(f"{'':>10} " + " ".join(str(l)[:cellw].ljust(cellw) for l in labels))
    return "\n".join(lines)


def _render_scatter(title: str, xvalues: List[float], values: List[float], unit: str) -> str:
    """Relationship: two numeric series as points on a grid."""
    xs, ys = [float(x) for x in xvalues], [float(y) for y in values]
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("scatter needs >= 2 (x, y) pairs: xvalues + values")
    W, H = 40, 10
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    if xhi == xlo:
        xhi = xlo + 1.0
    if yhi == ylo:
        yhi = ylo + 1.0
    grid = [[" "] * W for _ in range(H)]
    for x, y in zip(xs, ys):
        c = round((x - xlo) / (xhi - xlo) * (W - 1))
        grid[H - 1 - round((y - ylo) / (yhi - ylo) * (H - 1))][c] = "●"
    yw = len(_fmt_amount(yhi, unit))
    lines = [f"{title} ({unit})"]
    for r in range(H):
        yv = ylo + (yhi - ylo) * (H - 1 - r) / (H - 1)
        lines.append(f"{_fmt_amount(yv, unit):>{yw}} ┤ " + "".join(grid[r]))
    lines.append(" " * (yw + 1) + "└" + "─" * W)
    left, right = _fmt_amount(xlo, unit), _fmt_amount(xhi, unit)
    lines.append(" " * (yw + 2) + left.ljust(W - len(right)) + right)
    return "\n".join(lines)


def make_chart(
    chart_type: str,
    title: str,
    labels: List[str] | None = None,
    values: List[float] | None = None,
    unit: str = "₹",
    xvalues: List[float] | None = None,
) -> Dict[str, Any]:
    """Render a to-scale chart as a markdown code block the model must paste verbatim."""
    if chart_type == "scatter":
        if not xvalues or not values:
            raise ValueError("scatter needs both xvalues and values")
        block = _render_scatter(title, xvalues, values, unit)
    else:
        renderers = {"bar": _render_bar, "line": _render_line,
                     "stacked": _render_stacked, "histogram": _render_histogram}
        if chart_type not in renderers:
            raise ValueError(
                f"unknown chart_type {chart_type!r}; use bar, line, stacked, histogram or scatter")
        if not labels or not values or len(labels) != len(values):
            raise ValueError("labels and values must be non-empty lists of the same length")
        block = renderers[chart_type](title, [str(l) for l in labels], list(values), unit)
    return {"markdown": f"```text\n{block}\n```"}


def _inline_chart_payloads(answer: str) -> str:
    """Safety net: if the model wrote chart definitions as ```json blocks in its
    answer instead of calling make_chart, render them properly anyway."""
    def repl(match: "re.Match[str]") -> str:
        try:
            payload = json.loads(match.group(1))
            if not isinstance(payload, dict) or "chart_type" not in payload:
                return match.group(0)
            return make_chart(**payload)["markdown"]
        except (json.JSONDecodeError, TypeError, ValueError):
            return match.group(0)
    return re.sub(r"```json\s*\n(.*?)```", repl, answer, flags=re.S)


# ---------------------------------------------------------------------------
# Terminal presentation: ANSI colours for display only - saved .md files stay
# clean, colours switch on automatically when stdout is a real terminal.
# ---------------------------------------------------------------------------
_COLOR = False
_ANSI = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "cyan": "\033[36m", "green": "\033[32m", "magenta": "\033[35m", "red": "\033[31m",
}


def _set_color(enabled: bool) -> None:
    global _COLOR
    _COLOR = enabled


def _c(code: str, text: str) -> str:
    return f"{code}{text}{_ANSI['reset']}" if _COLOR else text


def _colorize_md(text: str) -> str:
    """ANSI-colour the markdown for the terminal: headings cyan, **Insights**
    magenta, ₹ amounts green, bullets and table headers emphasised. Chart code
    blocks are left untouched so the character alignment stays exact."""
    if not _COLOR:
        return text
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    styled = []
    for part in parts:
        if part.startswith("```"):
            styled.append(part)
            continue
        new_lines = []
        lines = part.split("\n")
        for i, line in enumerate(lines):
            if re.match(r"#{1,6} ", line):
                new_lines.append(_c(_ANSI["bold"] + _ANSI["cyan"], line))
                continue
            if line.lstrip().startswith("**Insights**"):
                new_lines.append(_c(_ANSI["bold"] + _ANSI["magenta"], line.replace("**", "")))
                continue
            if re.match(r"^\s*[*-] ", line):
                line = re.sub(r"^(\s*)[*-] ", rf"\1{_c(_ANSI['magenta'], '•')} ", line)
            table_header = line.startswith("|") and i + 1 < len(lines) and re.match(
                r"^\|[\s:|-]+\|?\s*$", lines[i + 1] or "")
            if table_header:
                new_lines.append(_c(_ANSI["bold"], line))
                continue
            line = re.sub(r"\*\*(.+?)\*\*", lambda m: _c(_ANSI["bold"], m.group(1)), line)
            line = re.sub(r"₹\s?[\d,]+(?:\.\d{1,2})?", lambda m: _c(_ANSI["green"], m.group(0)), line)
            new_lines.append(line)
        styled.append("\n".join(new_lines))
    return "".join(styled)


# ---------------------------------------------------------------------------
# Rich markdown for viewers: inline-HTML colour spans that render in PyCharm,
# VSCode and Typora previews. (GitHub strips the style attribute but keeps the
# text, so nothing breaks there either.)
# ---------------------------------------------------------------------------
_RICH_COLORS = {"green": "#98c379", "purple": "#c678dd", "blue": "#61afef", "red": "#e06c75"}


def _rich_md(text: str) -> str:
    """Colour the markdown for .md viewers: ₹ amounts green, the Insights
    label purple-bold. Chart code blocks are left untouched so the character
    alignment stays exact."""
    def span(color: str, t: str) -> str:
        return f'<span style="color:{_RICH_COLORS[color]}">{t}</span>'

    parts = re.split(r"(```.*?```)", text, flags=re.S)
    styled = []
    for part in parts:
        if part.startswith("```"):
            styled.append(part)
            continue
        lines = part.split("\n")
        for i, line in enumerate(lines):
            if line.strip() == "**Insights**":
                lines[i] = f'<b>{span("purple", "Insights")}</b>'
            else:
                lines[i] = re.sub(r"₹\s?[\d,]+(?:\.\d{1,2})?",
                                  lambda m: span("green", m.group(0)), line)
        styled.append("\n".join(lines))
    return "".join(styled)


TOOL_FUNCS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "list_tables": list_tables,
    "describe_table": describe_table,
    "run_sql": run_sql,
    "make_chart": make_chart,
}

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List all table names in the database.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": "Columns, types, primary/foreign keys and 3 sample rows for one table.",
            "parameters": {
                "type": "object",
                "properties": {"table": {"type": "string", "description": "Table name from list_tables."}},
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Run ONE read-only SELECT/WITH query against the database. "
                "LIMIT is added automatically and results are capped at 50 rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string", "description": "The SQL query to run."}},
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_chart",
            "description": (
                "Render a to-scale chart as a markdown code block to paste verbatim. "
                "Pick chart_type by what the data shows: 'bar' = comparing/ranking categories; "
                "'line' = trend over time; 'stacked' = part-to-whole composition (share of total); "
                "'histogram' = distribution (counts per bin); 'scatter' = relationship between "
                "two numeric series (pass xvalues). Use 1-2 charts, only when they add value."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line", "stacked", "histogram", "scatter"],
                        "description": "bar: ranking; line: trend over time; stacked: composition share; histogram: distribution; scatter: two-numeric relationship.",
                    },
                    "title": {"type": "string", "description": "Chart title, e.g. 'Monthly spend trend'."},
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Row/bin labels in display order (keep them short, e.g. 'Sep', '0-500'). Ignored for scatter.",
                    },
                    "values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Numbers exactly as returned by run_sql (y values for scatter).",
                    },
                    "unit": {"type": "string", "description": "Unit prefix. Defaults to '₹'."},
                    "xvalues": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Scatter only: numeric x coordinates.",
                    },
                },
                "required": ["chart_type", "title", "values"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Ollama plumbing (identical to ollama-agent/agent.py)
# ---------------------------------------------------------------------------


class OllamaError(RuntimeError):
    """Raised when Ollama cannot be reached or returns an error."""


def chat(
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    base_url: str = OLLAMA_URL,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": 16384},
    }
    request = urllib.request.Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"could not reach Ollama at {base_url} ({exc.reason}). Is `ollama serve` running?"
        ) from exc


def run_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    func = TOOL_FUNCS.get(name)
    if func is None:
        return {"error": f"unknown tool {name!r}"}
    try:
        return func(**arguments)
    except TypeError as exc:
        return {"error": f"bad arguments for {name!r}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - errors go back to the model, not the user
        return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def run_agent(
    prompt: str,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_URL,
    system: str | None = None,
    audit: bool = True,
) -> str:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system or build_system_prompt()},
        {"role": "user", "content": prompt},
    ]
    audited = False

    for step in range(1, MAX_STEPS + 1):
        response = chat(model, messages, TOOL_SCHEMAS, base_url=base_url)
        message = response.get("message") or {}
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            content = (message.get("content") or "").strip() or "(model returned no content)"
            if not audit or audited:
                print("\n" + _c(_ANSI["bold"] + _ANSI["green"], f"=== final answer (step {step}) ==="))
                return _inline_chart_payloads(content)
            # self-audit: one extra pass to re-verify every derived figure with SQL
            audited = True
            print("\n" + _c(_ANSI["bold"] + _ANSI["cyan"],
                            f"=== draft answer (step {step}) — auditing figures ==="))
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": AUDIT_PROMPT})
            continue

        print(f"=== step {step} ===")
        messages.append(
            {"role": "assistant", "content": message.get("content") or "", "tool_calls": tool_calls}
        )
        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name", "")
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):  # some models emit JSON as a string
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            result = run_tool(name, arguments)
            printable = json.dumps(result, ensure_ascii=False)
            print(_c(_ANSI["cyan"], f"--> {name}({json.dumps(arguments, ensure_ascii=False)})"))
            result_line = f"    = {printable[:300]}{'…' if len(printable) > 300 else ''}"
            print(_c(_ANSI["red"] if '"error"' in printable else _ANSI["dim"], result_line))
            messages.append({"role": "tool", "name": name, "content": printable})

    return "(stopped after MAX_STEPS without a final answer)"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only SQL data-explorer agent (Ollama).")
    parser.add_argument("question", nargs="?", default="Which product category earns the most revenue?",
                        help="The question to answer from the database.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--list-tools", action="store_true", help="Print the tool schemas and exit.")
    parser.add_argument("--save", action="store_true", help="Also write the final markdown report to reports/.")
    parser.add_argument("--no-audit", action="store_true", help="Skip the figure-audit pass (faster, slightly less reliable).")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colours in terminal output.")
    parser.add_argument("--base-url", default=OLLAMA_URL, help="Ollama /api/chat endpoint.")
    args = parser.parse_args(argv)

    if args.list_tools:
        print(json.dumps(TOOL_SCHEMAS, indent=2))
        return 0

    print(f"model: {args.model}")
    print(f"question: {args.question}")
    _set_color((not args.no_color) and sys.stdout.isatty() and not os.environ.get("NO_COLOR"))
    try:
        answer = run_agent(args.question, model=args.model, base_url=args.base_url,
                           audit=not args.no_audit)
    except OllamaError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    print(_colorize_md(answer))
    if args.save:
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        path = os.path.join(reports_dir, f"report-{datetime.now():%Y%m%d-%H%M%S}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# Spendly report\n\n**Question:** {args.question}\n\n{_rich_md(answer)}\n")
        print(f"\nsaved: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
