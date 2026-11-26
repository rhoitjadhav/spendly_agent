#!/usr/bin/env python3
"""CLI for the spendly SQL agent.

Usage
-----
    python3 src/make_demo_db.py                              # once, to build spendly.db
    python3 src/main.py "How much did I spend on Food & Dining in October?"
    python3 src/main.py "Total spend per category" --save   # colourful report in reports/*.md
    python3 src/main.py --model llama3.2:latest "Total spend per source?"   # faster but weaker
    python3 src/main.py --list-tools
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

from agent import SqlAgent
from constants import (
    DEFAULT_MODEL,
    DEFAULT_QUESTION,
    OLLAMA_HOST,
    REPORTS_DIR,
    TOOL_SCHEMAS,
)
from utils import colorize_md, rich_md, set_color


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only SQL data-explorer agent (Ollama).")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION,
                        help="The question to answer from the database.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--list-tools", action="store_true", help="Print the tool schemas and exit.")
    parser.add_argument("--save", action="store_true", help="Also write the final markdown report to reports/.")
    parser.add_argument("--no-audit", action="store_true", help="Skip the figure-audit pass (faster, slightly less reliable).")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colours in terminal output.")
    parser.add_argument("--host", default=OLLAMA_HOST, help="Ollama host (default: %(default)s).")
    args = parser.parse_args(argv)

    if args.list_tools:
        print(json.dumps(TOOL_SCHEMAS, indent=2))
        return 0

    print(f"model: {args.model}")
    print(f"question: {args.question}")
    set_color((not args.no_color) and sys.stdout.isatty() and not os.environ.get("NO_COLOR"))
    answer = SqlAgent(model=args.model, host=args.host,
                      audit=not args.no_audit).run(args.question)
    print(colorize_md(answer))
    if args.save:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        path = os.path.join(REPORTS_DIR, f"report-{datetime.now():%Y%m%d-%H%M%S}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# Spendly report\n\n**Question:** {args.question}\n\n{rich_md(answer)}\n")
        print(f"\nsaved: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
