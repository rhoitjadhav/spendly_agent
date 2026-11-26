#!/usr/bin/env python3
"""Helper functions for the spendly agent: chart rendering, markdown colouring
(ANSI for the terminal, inline-HTML spans for .md viewers) and system-prompt
construction. Everything here is pure presentation or prompt plumbing - no
Ollama, no database access."""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any

from constants import ANSI, CHART_CHARS, RICH_COLORS, SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


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
# Chart rendering (to scale, as markdown code blocks)
# ---------------------------------------------------------------------------


def fmt_amount(v: float, unit: str = "₹") -> str:
    """Compact human format for chart axes: ₹3.85L / ₹106.2k / ₹649."""
    v = float(v)
    if abs(v) >= 1e5:
        return f"{unit}{v / 1e5:,.2f}L"
    if abs(v) >= 1e3:
        return f"{unit}{v / 1e3:,.1f}k"
    if v.is_integer():
        return f"{unit}{v:,.0f}"
    return f"{unit}{v:,.2f}"


def render_bar(title: str, labels: list[str], values: list[float], unit: str) -> str:
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


def render_line(title: str, labels: list[str], values: list[float], unit: str) -> str:
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
    yw = len(fmt_amount(hi, unit))
    lines = [f"{title} ({unit})"]
    for r in range(rows_n):
        level = lo + (hi - lo) * (rows_n - 1 - r) / (rows_n - 1)
        cells = "".join(c.ljust(cellw) for c in grid[r])
        lines.append(f"{fmt_amount(level, unit):>{yw}} ┤ {cells}")
    lines.append(" " * (yw + 1) + "└" + "─" * (len(vals) * (cellw + 1) - 1))
    lines.append(" " * (yw + 2) + " ".join(str(l)[:cellw].ljust(cellw) for l in labels))
    return "\n".join(lines)


def render_stacked(title: str, labels: list[str], values: list[float], unit: str) -> str:
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


def render_histogram(title: str, labels: list[str], values: list[float], unit: str) -> str:
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


def render_scatter(title: str, xvalues: list[float], values: list[float], unit: str) -> str:
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
    yw = len(fmt_amount(yhi, unit))
    lines = [f"{title} ({unit})"]
    for r in range(H):
        yv = ylo + (yhi - ylo) * (H - 1 - r) / (H - 1)
        lines.append(f"{fmt_amount(yv, unit):>{yw}} ┤ " + "".join(grid[r]))
    lines.append(" " * (yw + 1) + "└" + "─" * W)
    left, right = fmt_amount(xlo, unit), fmt_amount(xhi, unit)
    lines.append(" " * (yw + 2) + left.ljust(W - len(right)) + right)
    return "\n".join(lines)


def make_chart(
    chart_type: str,
    title: str,
    labels: list[str] | None = None,
    values: list[float] | None = None,
    unit: str = "₹",
    xvalues: list[float] | None = None,
) -> dict[str, Any]:
    """Render a to-scale chart as a markdown code block the model must paste verbatim."""
    if chart_type == "scatter":
        if not xvalues or not values:
            raise ValueError("scatter needs both xvalues and values")
        block = render_scatter(title, xvalues, values, unit)
    else:
        renderers = {"bar": render_bar, "line": render_line,
                     "stacked": render_stacked, "histogram": render_histogram}
        if chart_type not in renderers:
            raise ValueError(
                f"unknown chart_type {chart_type!r}; use bar, line, stacked, histogram or scatter")
        if not labels or not values or len(labels) != len(values):
            raise ValueError("labels and values must be non-empty lists of the same length")
        block = renderers[chart_type](title, [str(l) for l in labels], list(values), unit)
    return {"markdown": f"```text\n{block}\n```"}


def inline_chart_payloads(answer: str) -> str:
    """Safety net: if the model wrote chart definitions as ```json blocks in its
    answer instead of calling make_chart, render them properly anyway."""
    def repl(match: re.Match[str]) -> str:
        try:
            payload = json.loads(match.group(1))
            if not isinstance(payload, dict) or "chart_type" not in payload:
                return match.group(0)
            return make_chart(**payload)["markdown"]
        except (json.JSONDecodeError, TypeError, ValueError):
            return match.group(0)
    return re.sub(r"```json\s*\n(.*?)```", repl, answer, flags=re.DOTALL)


# ---------------------------------------------------------------------------
# Terminal presentation: ANSI colours for display only - saved .md files stay
# clean, colours switch on automatically when stdout is a real terminal.
# ---------------------------------------------------------------------------
_COLOR = False


def set_color(enabled: bool) -> None:
    global _COLOR
    _COLOR = enabled


def c(code: str, text: str) -> str:
    return f"{code}{text}{ANSI['reset']}" if _COLOR else text


def colorize_md(text: str) -> str:
    """ANSI-colour the markdown for the terminal: headings cyan, **Insights**
    magenta, ₹ amounts green, bullets and table headers emphasised. Chart code
    blocks are left untouched so the character alignment stays exact."""
    if not _COLOR:
        return text
    parts = re.split(r"(```.*?```)", text, flags=re.DOTALL)
    styled = []
    for part in parts:
        if part.startswith("```"):
            styled.append(part)
            continue
        new_lines = []
        lines = part.split("\n")
        for i, line in enumerate(lines):
            if re.match(r"#{1,6} ", line):
                new_lines.append(c(ANSI["bold"] + ANSI["cyan"], line))
                continue
            if line.lstrip().startswith("**Insights**"):
                new_lines.append(c(ANSI["bold"] + ANSI["magenta"], line.replace("**", "")))
                continue
            if re.match(r"^\s*[*-] ", line):
                line = re.sub(r"^(\s*)[*-] ", rf"\1{c(ANSI['magenta'], '•')} ", line)
            table_header = line.startswith("|") and i + 1 < len(lines) and re.match(
                r"^\|[\s:|-]+\|?\s*$", lines[i + 1] or "")
            if table_header:
                new_lines.append(c(ANSI["bold"], line))
                continue
            line = re.sub(r"\*\*(.+?)\*\*", lambda m: c(ANSI["bold"], m.group(1)), line)
            line = re.sub(r"₹\s?[\d,]+(?:\.\d{1,2})?", lambda m: c(ANSI["green"], m.group(0)), line)
            new_lines.append(line)
        styled.append("\n".join(new_lines))
    return "".join(styled)


# ---------------------------------------------------------------------------
# Rich markdown for viewers: inline-HTML colour spans that render in PyCharm,
# VSCode and Typora previews. (GitHub strips the style attribute but keeps the
# text, so nothing breaks there either.)
# ---------------------------------------------------------------------------


def rich_md(text: str) -> str:
    """Colour the markdown for .md viewers: ₹ amounts green, the Insights
    label purple-bold. Chart code blocks are left untouched so the character
    alignment stays exact."""
    def span(color: str, t: str) -> str:
        return f'<span style="color:{RICH_COLORS[color]}">{t}</span>'

    parts = re.split(r"(```.*?```)", text, flags=re.DOTALL)
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
