#!/usr/bin/env python3
"""Shared constants for the spendly agent: Ollama settings, prompts, database
limits, SQL-guard patterns and the colour palettes used by utils.py."""
from __future__ import annotations

import os
import re

# The project root (the directory containing src/); spendly.db and reports/
# live there, not inside src/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Agent / Ollama ---------------------------------------------------------
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:latest"  # most reliable for multi-join SQL; llama3.2:latest is faster but weaker
MAX_STEPS = 14  # explore + query + chart + audit pass all need room
TEMPERATURE = 0.0  # deterministic: figures must be reproducible
NUM_CTX = 16384  # context window the model sees (system prompt + tool results)
DEFAULT_QUESTION = "Which product category earns the most revenue?"

# --- Reports (--save) ---------------------------------------------------------
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# --- Database access (tools.py) ----------------------------------------------
DB_PATH = os.environ.get(
    "SQLAGENT_DB", os.path.join(PROJECT_ROOT, "spendly.db")
)
MAX_ROWS = 50
MAX_CELL_CHARS = 200
QUERY_TIMEOUT_SECONDS = 10.0

# SQL guard patterns (tools.py): one read-only SELECT/WITH, and it must carry a LIMIT.
FORBIDDEN_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|grant|revoke|"
    r"attach|detach|pragma|vacuum|reindex|into)\b",
    re.IGNORECASE,
)
SELECTISH_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
LIMIT_RE = re.compile(r"\blimit\s+\d+\b", re.IGNORECASE)

# --- Tool schemas (what the model sees) ----------------------------------------
TOOL_SCHEMAS: list[dict] = [
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
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Table name from list_tables.",
                    }
                },
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
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL query to run.",
                    }
                },
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
                    "title": {
                        "type": "string",
                        "description": "Chart title, e.g. 'Monthly spend trend'.",
                    },
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
                    "unit": {
                        "type": "string",
                        "description": "Unit prefix. Defaults to '₹'.",
                    },
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

# --- Prompts ----------------------------------------------------------------
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

# --- Charts (utils.py) --------------------------------------------------------
CHART_CHARS = "█▓▒░▄▀●■▲◆"  # distinct fills for stacked segments

# ANSI colours for terminal display only - saved .md files stay clean.
ANSI = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "cyan": "\033[36m", "green": "\033[32m", "magenta": "\033[35m", "red": "\033[31m",
}

# Inline-HTML colours for .md viewers (PyCharm, VSCode, Typora previews).
RICH_COLORS = {"green": "#98c379", "purple": "#c678dd", "blue": "#61afef", "red": "#e06c75"}
