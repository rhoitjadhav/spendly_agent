# Spendly Agent

A read-only SQL data-explorer agent that answers questions about a personal
expense-tracker database, running entirely on a **local Ollama model** — no
cloud APIs, no API keys, no data leaving your machine.

You ask a question in plain English; the agent explores the schema, writes and
runs its own SQL, renders to-scale text charts, audits its own figures, and
returns a formatted analyst briefing:

```
python3 agent.py "How much did I spend on Food & Dining in October?"

Your total spending on Food & Dining in October was ₹6,402.09 📈. The highest
spending merchant in this category during October was Zomato, where you spent
₹2,915.48.

| Metric                     | Value (₹) |
| :------------------------- | :-------- |
| Total Food & Dining (Oct)  | 6,402.09  |
| Top Merchant (Oct)         | Zomato    |
| Top Merchant Spend         | 2,915.48  |

**Insights**
💡 Zomato accounted for a significant portion of your Food & Dining spend in October.
```

## How it works

The agent runs a standard tool-calling loop against Ollama's `/api/chat`
endpoint, with four tools:

| Tool | What it does |
| ---- | ------------ |
| `list_tables` | List table names in the database |
| `describe_table` | Columns, keys and 3 sample rows for one table |
| `run_sql` | Run **one guarded read-only** SELECT/WITH query (see below) |
| `make_chart` | Render a to-scale text chart (bar / line / stacked / histogram / scatter) as a markdown code block |

The loop, in order:

1. The model calls `list_tables`, then `describe_table` for the tables it needs.
2. It writes SQL, aggregates in the database (`SUM`, `GROUP BY`, …), and reads
   back the results.
3. It calls `make_chart` with values copied **exactly** from query results —
   the chart's proportions and shares are computed in code, never by the model.
4. Before the final answer, a **self-audit pass** re-verifies every derived
   figure (sums, shares, percentages) against the SQL results and recomputes
   anything it cannot confirm.

A few deliberate design choices keep the model honest:

- **Relative dates are resolved in Python, not by the model.** The system
  prompt is built with today's date, "this month" and "last month" already
  resolved to `YYYY-MM` strings, so the model never does calendar arithmetic.
- **The system prompt forbids invented numbers.** Every figure must come from
  a `run_sql` result in the conversation.
- **Charts are tool output, not model prose.** The model must paste the
  rendered code blocks verbatim; a safety net also re-renders any chart
  definition the model mistakenly wrote as inline JSON.

## Read-only by construction

`db.py` guards every query the model can make:

- Only a single `SELECT` / `WITH` statement is accepted.
- Write keywords (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, …) are
  rejected — even inside `WITH … INSERT` tricks — while ignoring keywords
  inside string literals.
- A `LIMIT` is appended when the model forgets one; results are capped at 50
  rows and long cell values are truncated.
- Queries are cut off after a 10-second timeout via a progress handler.

## Files

| File | Purpose |
| ---- | ------- |
| `agent.py` | The agent: tools, chart renderers, Ollama plumbing, CLI |
| `db.py` | Read-only database access: schema introspection + guarded SQL |
| `make_demo_db.py` | Seed `spendly.db` with ~3 months of dummy UPI/card transactions |
| `test_all.py` | Tests for the guard, introspection and agent loop (no network needed) |
| `spendly.db` | The seeded SQLite demo database |
| `reports/` | Saved markdown reports (written by `--save`) |

## Getting started

Requires Python 3.10+ and [Ollama](https://ollama.com) running locally
(`ollama serve`).

```bash
# 1. pull a model
ollama pull gemma4:latest        # default; most reliable for multi-join SQL

# 2. build the demo database (once)
python3 make_demo_db.py

# 3. ask a question
python3 agent.py "How much did I spend on Food & Dining in October?"
python3 agent.py "Total spend per category" --save    # also write reports/<timestamp>.md
```

The runtime uses only the Python standard library (sqlite3, urllib) — nothing
to install. `requirements.txt` lists optional extras (SQLAlchemy, MCP,
psycopg2) for building the DB from real models or targeting a real Postgres.

### CLI options

```
python3 agent.py [question] [--model MODEL] [--save] [--no-audit]
                       [--no-color] [--base-url URL] [--list-tools]
```

| Flag | Effect |
| ---- | ------ |
| `--model` | Ollama model (default `gemma4:latest`; `llama3.2:latest` is faster but weaker at SQL) |
| `--save` | Also write the final markdown report to `reports/` |
| `--no-audit` | Skip the figure-audit pass (faster, slightly less reliable) |
| `--no-color` | Disable ANSI colours in terminal output |
| `--base-url` | Override the Ollama endpoint (default `http://localhost:11434/api/chat`) |
| `--list-tools` | Print the tool JSON schemas and exit |

## Pointing it at your real database

`db.py` has two marked **SWAP POINTs**:

1. `get_connection()` — the single place to swap the bundled SQLite file for
   your real database (Postgres/MySQL examples are in its docstring). Best
   practice: use a dedicated **read-only database user** so that even a bug in
   the guard could never write to your data.
2. `list_tables()` / `describe_table()` — schema introspection; the
   `information_schema` queries for Postgres/MySQL are in the comments.

The database file location can also be overridden without editing code:

```bash
SQLAGENT_DB=/path/to/your.db python3 agent.py "..."
```

## Tests

```bash
python3 test_all.py
```

Part 1 exercises the database layer (guard, introspection, query execution)
against `spendly.db`; part 2 runs the agent loop with a scripted fake LLM —
both without any network access.
