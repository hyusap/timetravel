"""Shared prompt/action helpers for Reverse Code Door agent training/eval."""

from __future__ import annotations

import re
from typing import Optional

from benchmarks.reverse_code_door import TemporalAction


SYSTEM_PROMPT = """You are an agent in a 1-D corridor:
  [start pos=0] --- [door pos=1] --- [pos=2] --- [oracle pos=3]

Goal: unlock the door at pos=1 with the secret 3-digit code shown by the oracle at pos=3.
Budget: 6 steps total. Linear navigation takes 7 steps - you MUST use branch to rewind.

You may think step-by-step, but your final line MUST be exactly:
  ACTION: <command>

Allowed commands:
  forward
  backward
  inspect
  unlock <code>
  branch <ago> <instruction>"""

CODE_PATTERN = re.compile(r"\b(\d{3})\b")
FORWARD_PATTERN = re.compile(r"\bforward\b")
BACKWARD_PATTERN = re.compile(r"\bbackward\b")
INSPECT_PATTERN = re.compile(r"\binspect\b")
UNLOCK_PATTERN = re.compile(r"\bunlock(?:\s+(\d{3}))?\b")
BRANCH_PATTERN = re.compile(r"\bbranch\s+(\d+)(?:\s+([^\n\r]+))?")


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
    if not text.strip():
        return None

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    action_line = None
    for line in lines:
        if line.lower().startswith("action:"):
            action_line = line.split(":", 1)[1].strip().lower()
    if action_line is None:
        # Fallback: use the last non-empty line as best-effort command text.
        action_line = lines[-1].lower()

    clean = action_line
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

    # Recovery mode: extract first valid command from noisy text.
    noisy = text.lower()
    m_branch = BRANCH_PATTERN.search(noisy)
    if m_branch is not None:
        ago = int(m_branch.group(1))
        instruction = (m_branch.group(2) or "").strip()
        return TemporalAction(kind="branch", ago=ago, instruction=instruction)

    m_unlock = UNLOCK_PATTERN.search(noisy)
    if m_unlock is not None:
        code = m_unlock.group(1) or ""
        return TemporalAction(command="unlock", unlock_code=code)

    if INSPECT_PATTERN.search(noisy):
        return TemporalAction(command="inspect")
    if BACKWARD_PATTERN.search(noisy):
        return TemporalAction(command="backward")
    if FORWARD_PATTERN.search(noisy):
        return TemporalAction(command="forward")
    return None


def infer_success(obs: dict) -> bool:
    return bool(obs["info"].get("command") == "unlock" and obs["reward"] > 0)


def format_action(action: TemporalAction) -> str:
    """Return a compact canonical action string for chat history."""
    if action.kind != "step":
        if action.kind == "branch":
            return f"ACTION: branch {action.ago} {action.instruction}".strip()
        return f"ACTION: {action.kind}"
    if action.command == "unlock":
        return f"ACTION: unlock {action.unlock_code or ''}".strip()
    return f"ACTION: {action.command}"
