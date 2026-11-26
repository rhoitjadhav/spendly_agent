#!/usr/bin/env python3
"""Tests for the spendly SQL agent. Run: python3 src/test_all.py

Part 1: SqlTools guard + introspection + query execution against spendly.db (no network).
Part 2: the agent loop with a scripted fake LLM (no network) - proves tool
        results are fed back into the conversation correctly.
"""
from __future__ import annotations

import sqlite3
from datetime import date
from types import SimpleNamespace

import agent
import utils
from constants import DB_PATH, MAX_ROWS
from tools import SqlTools

# ---------------------------------------------------------------------------
# Part 1: the database layer
# ---------------------------------------------------------------------------

tools = SqlTools()

tables = tools.list_tables()["tables"]
assert tables == ["transactions"], tables

desc = tools.describe_table("transactions")
cols = [c["name"] for c in desc["columns"]]
assert cols == ["id", "source", "date", "description", "merchant", "amount",
                "category", "account", "note", "raw_text", "fingerprint",
                "created_at", "updated_at"], cols
assert desc["sample_rows"], "expected sample rows"

try:
    tools.describe_table("nope")
    raise AssertionError("unknown table should raise")
except ValueError:
    pass

con = sqlite3.connect(DB_PATH)
ground_truth = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
category_truth = dict(con.execute(
    "SELECT category, COUNT(*) FROM transactions GROUP BY category").fetchall())
con.close()

assert ground_truth > 300, f"expected a useful seed volume, got {ground_truth}"

# real query works, LIMIT gets appended
r = tools.run_sql("SELECT COUNT(*) AS n FROM transactions")
assert r["rows"][0]["n"] == ground_truth, (r, ground_truth)
assert r["sql"].endswith(f"LIMIT {MAX_ROWS}")

# typical analyst question against the seeded schema
r = tools.run_sql("SELECT category, COUNT(*) AS n FROM transactions GROUP BY category ORDER BY n DESC")
assert r["row_count"] == 10, r["row_count"]
assert sum(row["n"] for row in r["rows"]) == ground_truth
for row in r["rows"]:
    assert category_truth[row["category"]] == row["n"]

# data invariants from the brief: amounts positive, known categories/sources
r = tools.run_sql("SELECT COUNT(*) AS bad FROM transactions WHERE amount <= 0")
assert r["rows"][0]["bad"] == 0
r = tools.run_sql(
    "SELECT COUNT(*) AS bad FROM transactions WHERE category NOT IN "
    "('Food & Dining','Transport','Groceries','Shopping','Entertainment','Health',"
    "'Utilities','Education','Finance','Uncategorized')")
assert r["rows"][0]["bad"] == 0
r = tools.run_sql(
    "SELECT COUNT(*) AS bad FROM transactions WHERE source NOT IN ('navi','phonepe','googlepay')")
assert r["rows"][0]["bad"] == 0

# the guard blocks everything that could write or escape
blocked = [
    "DELETE FROM transactions",
    "UPDATE transactions SET amount = 0",
    "INSERT INTO transactions (source) VALUES ('navi')",
    "DROP TABLE transactions",
    "SELECT 1; DROP TABLE transactions",                  # multiple statements
    "WITH x AS (SELECT 1) INSERT INTO transactions SELECT * FROM x",  # writable CTE
    "PRAGMA table_info(transactions)",
    "SELECT * FROM transactions INTO backup",             # SELECT ... INTO
]
for sql in blocked:
    try:
        tools.run_sql(sql)
        raise AssertionError(f"guard should have blocked: {sql}")
    except ValueError:
        pass

# keywords inside string literals are fine (no false positives)
r = tools.run_sql("SELECT COUNT(*) AS n FROM transactions WHERE merchant = 'Delete Me Pvt Ltd'")
assert r["rows"][0]["n"] >= 0
print(f"part 1 ok: db layer ({ground_truth} transactions in spendly.db)")

# ---------------------------------------------------------------------------
# Part 2: the agent loop with a fake LLM (proves the conversation plumbing)
# ---------------------------------------------------------------------------


class FakeChat:
    """Scripted stand-in for llm.LLM. One response per agent step.
    Returns attribute-style objects like the real ollama package does."""

    def __init__(self, script):
        self.script, self.seen = script, []

    def run(self, messages, tools, **kwargs):
        self.seen.append([dict(m) for m in messages])
        return self.script[min(len(self.seen), len(self.script)) - 1]


def fake_agent(script) -> agent.SqlAgent:
    """A SqlAgent whose LLM is a scripted fake (no network)."""
    inst = agent.SqlAgent(model="fake")
    inst.llm = FakeChat(script)
    return inst


def tool_call(name, arguments):
    function = SimpleNamespace(name=name, arguments=arguments)
    call = SimpleNamespace(function=function)
    return SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[call]))


def final(text):
    return SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=[]))


# two-turn script: model runs a query, sees the result, answers
fake = fake_agent([
    tool_call("run_sql", {"sql": "SELECT COUNT(*) AS n FROM transactions"}),
    final(f"You have {ground_truth} transactions."),
])
answer = fake.run("How many transactions do I have?")
assert f"{ground_truth} transactions" in answer, answer

# the tool result must have been fed back as a 'tool' message before the final step
tool_msgs = [m for m in fake.llm.seen[1] if m.get("role") == "tool"]
assert tool_msgs, "tool result was not fed back"
assert f'"n": {ground_truth}' in tool_msgs[0]["content"], tool_msgs[0]["content"]

# a guarded query must come back to the model as an error message it can read
fake2 = fake_agent([
    tool_call("run_sql", {"sql": "DELETE FROM transactions"}),
    final("I cannot do that."),
])
fake2.run("delete everything")
tool_msgs2 = [m for m in fake2.llm.seen[1] if m.get("role") == "tool"]
assert tool_msgs2 and "read-only" in tool_msgs2[0]["content"], tool_msgs2

# the audit pass gives the model one extra round to fix derived figures
fake3 = fake_agent([
    tool_call("run_sql", {"sql": "SELECT COUNT(*) AS n FROM transactions"}),
    final("Draft: 333 transactions, total spend 999."),
    tool_call("run_sql", {"sql": "SELECT ROUND(SUM(amount), 2) AS total FROM transactions"}),
    final("Corrected: 333 transactions, total spend ₹384,765.60."),
])
answer3 = fake3.run("Summary?")
assert "Corrected" in answer3, answer3
audit_user_msgs = [m for m in fake3.llm.seen[2] if m.get("role") == "user"]
assert any("Audit" in m.get("content", "") for m in audit_user_msgs), "audit prompt not injected"

print("part 2 ok: agent loop plumbing")

# ---------------------------------------------------------------------------
# Part 3: chart rendering — all types deterministic and to scale
# ---------------------------------------------------------------------------

# bar: comparison/ranking, to scale
md = utils.make_chart("bar", "Spend by category", ["Shopping", "Food & Dining"], [200, 100])["markdown"]
assert md.startswith("```text") and md.rstrip().endswith("```")
bar_lines = [line for line in md.splitlines() if "█" in line]
assert len(bar_lines) == 2, md
w1, w2 = (line.count("█") for line in bar_lines)
assert w1 == 2 * w2 and "₹200" in bar_lines[0] and "₹100" in bar_lines[1], (w1, w2)

# stacked: shares computed by code and summing to 100%
md = utils.make_chart("stacked", "Share of spend", ["A", "B"], [200, 100])["markdown"]
assert "66.7%" in md and "33.3%" in md and "▓" in md, md

# line: one point per value, labels on the axis
md = utils.make_chart("line", "Monthly trend", ["Sep", "Oct", "Nov"], [100, 300, 200])["markdown"]
assert md.count("●") == 3 and "Sep" in md and "Nov" in md, md

# histogram: rendered blocks equal the sum of to-scale heights (8+4+2)
md = utils.make_chart("histogram", "Amount distribution", ["0-500", "500-1k", "1k+"], [10, 5, 2])["markdown"]
assert md.count("█") == 14, md

# scatter: one point per (x, y) pair
md = utils.make_chart("scatter", "Rel", ["x", "y", "z"], [1, 2, 3], xvalues=[10, 20, 30])["markdown"]
assert md.count("●") == 3, md

# bad inputs raise
for bad in [("pie", "T", ["a"], [1]), ("bar", "T", ["a", "b"], [1])]:
    try:
        utils.make_chart(bad[0], bad[1], bad[2], bad[3])
        raise AssertionError(f"should raise: {bad}")
    except ValueError:
        pass

# a model that inlines chart JSON as text still gets real charts (safety net)
draft = utils.inline_chart_payloads(
    'Intro\n```json\n{"chart_type": "stacked", "title": "Share", '
    '"labels": ["A", "B"], "values": [200, 100]}\n```\nOutro')
assert "66.7%" in draft and "```json" not in draft and "█" in draft, draft
# non-chart json blocks pass through untouched
untouched = utils.inline_chart_payloads('```json\n{"foo": 1}\n```')
assert '"foo": 1' in untouched

print("part 3 ok: chart rendering")

# ---------------------------------------------------------------------------
# Part 4: terminal colours (display-only; saved .md stays clean)
# ---------------------------------------------------------------------------

sample = "### Chart\n₹1,234.56 total\n**Insights**\n* ₹5 note\n```text\n₹99 ┤ █\n```"
utils.set_color(False)
assert utils.colorize_md(sample) == sample, "colours must stay off when disabled"

utils.set_color(True)
colored = utils.colorize_md(sample)
assert "\033[36m### Chart" in colored, colored                       # cyan heading
assert "\033[35m•" in colored, colored                               # magenta bullet
assert "\033[1m\033[35mInsights\033[0m" in colored, colored          # bold+magenta Insights
assert "₹1,234.56" in colored and "\033[32m₹1,234.56\033[0m" in colored
code_block = colored.split("```")[1]
assert "\033[" not in code_block, code_block                          # charts stay clean
utils.set_color(False)

print("part 4 ok: terminal colours")

# ---------------------------------------------------------------------------
# Part 5: rich markdown for viewers (inline-HTML colour spans)
# ---------------------------------------------------------------------------

sample = ("Total ₹1,234.56 spent\n\n| Category | Amount |\n| --- | --- |\n"
          "| Food | ₹5 |\n\n**Insights**\n* ok\n\n```text\n₹99 ┤ █\n```")
rich = utils.rich_md(sample)
assert '<span style="color:#98c379">₹1,234.56</span>' in rich, rich       # green amounts
assert '<span style="color:#98c379">₹5</span>' in rich, rich              # ...in tables too
assert '<b><span style="color:#c678dd">Insights</span></b>' in rich, rich  # purple label
chart = rich.split("```")[1]
assert "<span" not in chart, chart                                         # charts untouched
assert "| --- | --- |" in rich, rich                                       # tables intact

print("part 5 ok: rich markdown for viewers")

# ---------------------------------------------------------------------------
# Part 6: date anchoring - 'last month' must resolve without the model guessing
# ---------------------------------------------------------------------------

prompt = utils.build_system_prompt(date(2026, 11, 24))
assert "Today is 2026-11-24" in prompt, prompt
assert '"This month" = 2026-11' in prompt and '"last month" = 2026-10' in prompt, prompt
# month arithmetic across a year boundary (Jan -> Dec)
prompt = utils.build_system_prompt(date(2027, 1, 9))
assert '"This month" = 2027-01' in prompt and '"last month" = 2026-12' in prompt, prompt
# run() injects it as the system message
fake4 = fake_agent([final("ok")])
fake4.run("x")
assert "Today is" in fake4.llm.seen[0][0]["content"], fake4.llm.seen[0][0]["content"][:200]

print("part 6 ok: date anchoring")
print("\nAll tests passed.")
