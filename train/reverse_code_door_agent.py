"""Shared prompt/action helpers for Reverse Code Door agent training/eval."""

from __future__ import annotations

import re
from typing import Optional

from benchmarks.reverse_code_door import TemporalAction


SYSTEM_PROMPT = """You are an agent in a 1-D corridor:
  [start pos=0] --- [door pos=1] --- [pos=2] --- [oracle pos=3]

Goal: unlock the door at pos=1 with the secret 3-digit code shown by the oracle at pos=3.
Budget: 6 steps total. Linear navigation takes 7 steps - you MUST use branch to rewind.

Output exactly one action per turn in this format:
  forward
  backward
  inspect
  unlock <code>
  branch <ago> <instruction>"""

CODE_PATTERN = re.compile(r"\b(\d{3})\b")


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


def parse_action(text: str) -> Optional[TemporalAction]:
    """Parse model output into a TemporalAction."""

    clean = text.strip().lower().splitlines()[0].strip() if text.strip() else ""
    if clean == "forward":
        return TemporalAction(command="forward")
    if clean == "backward":
        return TemporalAction(command="backward")
    if clean == "inspect":
        return TemporalAction(command="inspect")
    if clean.startswith("unlock"):
        parts = clean.split()
        code = parts[1] if len(parts) > 1 else ""
        code_match = CODE_PATTERN.search(code)
        parsed_code = code_match.group(1) if code_match else code
        return TemporalAction(command="unlock", unlock_code=parsed_code)
    if clean.startswith("branch"):
        parts = clean.split(None, 2)
        try:
            ago = int(parts[1])
            instruction = parts[2] if len(parts) > 2 else ""
            return TemporalAction(kind="branch", ago=ago, instruction=instruction)
        except (IndexError, ValueError):
            return None
    return None


def infer_success(obs: dict) -> bool:
    return bool(obs["info"].get("command") == "unlock" and obs["reward"] > 0)
