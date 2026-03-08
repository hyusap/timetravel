"""Shared prompt/action helpers for Reverse Code Door agent training/eval."""

from __future__ import annotations

import json
import re
from typing import Optional

from benchmarks.reverse_code_door import TemporalAction


SYSTEM_PROMPT = """You are an agent in a 1-D corridor:
  [start pos=0] --- [door pos=1] --- [pos=2] --- [oracle pos=3]

Goal: unlock the door at pos=1 with the secret 3-digit code shown by the oracle at pos=3.
Budget: 6 steps total. Linear navigation takes 7 steps, so branching is required.

Output exactly one JSON object and nothing else.
Always include both:
  "thinking": brief reasoning
  "command": one valid command

Valid formats:
  {"thinking":"...","command":"forward"}
  {"thinking":"...","command":"backward"}
  {"thinking":"...","command":"inspect"}
  {"thinking":"...","command":"unlock","code":"123"}
  {"thinking":"...","command":"branch","ago":2,"instruction":"Use code 123 at door"}
"""

CODE_PATTERN = re.compile(r"\b(\d{3})\b")
JSON_CANDIDATE_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)


def obs_to_text(obs: dict, step_num: int) -> str:
    return "\n".join(
        [
            f"Step {step_num} | Budget remaining: {obs['remaining_budget']}",
            f"Position: {obs['position']} | At door: {obs['at_door']} | At oracle: {obs['at_oracle']}",
            f"Visible code: {obs['visible_code'] or 'None'}",
            f"Instruction hint: {obs['instruction_hint'] or 'None'}",
            f"Last branch: {obs['last_branch_event']['ago'] if obs['last_branch_event'] else 'None'}",
        ]
    )


def _parse_action_dict(payload: dict) -> Optional[TemporalAction]:
    command = str(payload.get("command", "")).strip().lower()
    if command in {"forward", "backward", "inspect", "wait"}:
        return TemporalAction(command=command)
    if command == "unlock":
        code = str(payload.get("code", "")).strip()
        m = CODE_PATTERN.search(code)
        code = m.group(1) if m else code
        return TemporalAction(command="unlock", unlock_code=code)
    if command == "branch":
        ago_val = payload.get("ago")
        try:
            ago = int(ago_val)
        except (TypeError, ValueError):
            return None
        if ago <= 0:
            return None
        instruction = str(payload.get("instruction", "")).strip()
        return TemporalAction(kind="branch", ago=ago, instruction=instruction)
    return None


def parse_action(text: str) -> Optional[TemporalAction]:
    """Parse model output JSON into a TemporalAction."""
    if not text.strip():
        return None

    for candidate in JSON_CANDIDATE_PATTERN.findall(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            action = _parse_action_dict(payload)
            if action is not None:
                return action

    # Fallback: full text may itself be a JSON object without clean regex capture.
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return _parse_action_dict(payload)
    except json.JSONDecodeError:
        return None

    return None


def infer_success(obs: dict) -> bool:
    return bool(obs["info"].get("command") == "unlock" and obs["reward"] > 0)


def format_action(action: TemporalAction) -> str:
    """Return a compact canonical JSON action for chat history."""
    if action.kind == "branch":
        return json.dumps(
            {
                "command": "branch",
                "ago": action.ago,
                "instruction": action.instruction or "",
            },
            separators=(",", ":"),
        )

    if action.kind != "step":
        return json.dumps({"command": action.kind}, separators=(",", ":"))

    if action.command == "unlock":
        return json.dumps({"command": "unlock", "code": action.unlock_code or ""}, separators=(",", ":"))

    return json.dumps({"command": action.command}, separators=(",", ":"))
