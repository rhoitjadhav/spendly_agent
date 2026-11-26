#!/usr/bin/env python3
"""A read-only SQL data-explorer agent on top of a local Ollama model.

The SqlAgent runs a tool-calling loop: the model thinks, calls tools
(list_tables / describe_table / run_sql / make_chart, see tools.py), sees
the results, and keeps going until it produces a final answer -- with one
self-audit pass that re-verifies every derived figure with SQL.

Layout: constants.py holds every constant (models, prompts, limits, colours),
utils.py the helpers (chart rendering, markdown colouring, prompt building),
tools.py the class-based tools, their schemas and the dispatcher, llm.py the
Ollama plumbing, main.py the CLI, and this file the agent class.
"""

from __future__ import annotations

import json
from typing import Any

from constants import (
    ANSI,
    AUDIT_PROMPT,
    DEFAULT_MODEL,
    MAX_STEPS,
    OLLAMA_HOST,
    TOOL_SCHEMAS,
)
from llm import LLM
from tools import SqlTools
from utils import build_system_prompt, c, inline_chart_payloads


class SqlAgent:
    """The agent loop: think -> call tools -> observe, until a final answer.

    All state the loop needs (llm client, tools, system prompt, audit) lives
    on the instance, so tests can stub `llm` and inject a scripted LLM.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = OLLAMA_HOST,
        tools: SqlTools | None = None,
        system: str | None = None,
        audit: bool = True,
    ):
        self.llm = LLM(model=model, host=host)
        self.tools = tools or SqlTools()
        self.system = system
        self.audit = audit

    def run(self, prompt: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system or build_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        audited = False

        for step in range(1, MAX_STEPS + 1):
            response = self.llm.run(messages, TOOL_SCHEMAS)
            message = response.message
            tool_calls = message.tool_calls or []

            if not tool_calls:
                content = (message.content or "").strip() or "(model returned no content)"
                if not self.audit or audited:
                    print("\n" + c(ANSI["bold"] + ANSI["green"], f"=== final answer (step {step}) ==="))
                    return inline_chart_payloads(content)
                # self-audit: one extra pass to re-verify every derived figure with SQL
                audited = True
                print("\n" + c(ANSI["bold"] + ANSI["cyan"],
                                f"=== draft answer (step {step}) — auditing figures ==="))
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": AUDIT_PROMPT})
                continue

            print(f"=== step {step} ===")
            messages.append(
                {"role": "assistant", "content": message.content or "", "tool_calls": tool_calls}
            )
            for call in tool_calls:
                function = call.function
                name = function.name
                arguments = function.arguments or {}
                if isinstance(arguments, str):  # some models emit JSON as a string
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                result = self.tools.run_tool(name, arguments)
                printable = json.dumps(result, ensure_ascii=False)
                print(c(ANSI["cyan"], f"--> {name}({json.dumps(arguments, ensure_ascii=False)})"))
                result_line = f"    = {printable[:300]}{'…' if len(printable) > 300 else ''}"
                print(c(ANSI["red"] if '"error"' in printable else ANSI["dim"], result_line))
                messages.append({"role": "tool", "name": name, "content": printable})

        return "(stopped after MAX_STEPS without a final answer)"
