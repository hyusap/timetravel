"""
Evaluate the base GPT-OSS model as an agent on ReverseCodeDoorEnv.
Run: uv run python train/eval_base.py
"""
import sys
import re
import torch
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from benchmarks.reverse_code_door import ReverseCodeDoorEnv, TemporalAction

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an agent in a 1-D corridor:
  [start pos=0] --- [door pos=1] --- [pos=2] --- [oracle pos=3]

Goal: unlock the door at pos=1 with the secret 3-digit code revealed by the oracle at pos=3.
Budget: 6 steps total. Walking to the oracle and back takes 7 steps — impossible without branching.
Use `branch <ago> <instruction>` to rewind the timeline after reading the code.

Output exactly one action per turn (nothing else):
  forward
  backward
  inspect
  unlock <3-digit-code>
  branch <ago> <instruction>
  abandon\
"""


def obs_to_text(obs: dict, step: int) -> str:
    return (
        f"Step {step} | Budget remaining: {obs['remaining_budget']}\n"
        f"Position: {obs['position']} | At door: {obs['at_door']} | At oracle: {obs['at_oracle']}\n"
        f"Visible code: {obs['visible_code'] or 'None'}\n"
        f"Instruction hint: {obs['instruction_hint'] or 'None'}\n"
        f"Last branch: {obs['last_branch_event']['ago'] if obs['last_branch_event'] else 'None'}"
    )


def parse_action(text: str) -> Optional[TemporalAction]:
    text = text.strip().lower().splitlines()[0].strip()
    if text == "forward":
        return TemporalAction(command="forward")
    if text == "backward":
        return TemporalAction(command="backward")
    if text == "inspect":
        return TemporalAction(command="inspect")
    if text == "abandon":
        return TemporalAction(kind="abandon")
    if text.startswith("unlock"):
        parts = text.split()
        return TemporalAction(command="unlock", unlock_code=parts[1] if len(parts) > 1 else "")
    if text.startswith("branch"):
        parts = text.split(None, 2)
        try:
            ago = int(parts[1])
            instruction = parts[2] if len(parts) > 2 else ""
            return TemporalAction(kind="branch", ago=ago, instruction=instruction)
        except (IndexError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Single episode rollout
# ---------------------------------------------------------------------------

def run_episode(model, tokenizer, seed: int, verbose: bool = False) -> dict:
    env = ReverseCodeDoorEnv()
    obs = env.reset(seed=seed)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    model.eval()
    with torch.inference_mode():
        for step in range(10):
            messages.append({"role": "user", "content": obs_to_text(obs, step + 1)})

            prompt_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)

            out = model.generate(
                prompt_ids,
                max_new_tokens=32,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            action_text = tokenizer.decode(
                out[0][prompt_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            action = parse_action(action_text) or TemporalAction(kind="abandon")
            messages.append({"role": "assistant", "content": action_text})

            obs = env.step(action)

            if verbose:
                kind = action.kind if action.kind != "step" else action.command
                print(f"  step {step+1} [{kind}] → pos={obs['position']} budget={obs['remaining_budget']} reward={obs['reward']:.3f}")

            if obs["done"]:
                break

    success = obs["info"].get("command") == "unlock" and obs["reward"] > 0
    used_branch = any(e["event_type"] == "branch" for e in env.meta_events)
    return {"success": success, "used_branch": used_branch, "steps": len(env.meta_events)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from unsloth import FastLanguageModel

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b",
        load_in_4bit=True,
        max_seq_length=1024,
    )

    # Verbose walkthrough of a single episode
    print("\n=== Single episode (seed=0) ===")
    env = ReverseCodeDoorEnv()
    obs = env.reset(seed=0)
    print(f"Secret code (hidden): {env._secret_code}\n")
    result = run_episode(model, tokenizer, seed=0, verbose=True)
    print(f"Result: success={result['success']}  used_branch={result['used_branch']}\n")

    # Bulk evaluation
    N = 50
    print(f"=== Bulk eval over {N} episodes ===")
    successes = 0
    branch_used = 0
    for seed in range(N):
        r = run_episode(model, tokenizer, seed=seed)
        successes += int(r["success"])
        branch_used += int(r["used_branch"])
        print(f"  seed={seed:3d}  success={r['success']}  branch={r['used_branch']}", flush=True)

    print(f"\nSuccess rate : {successes}/{N} = {successes/N:.0%}")
    print(f"Branch rate  : {branch_used}/{N} = {branch_used/N:.0%}")


if __name__ == "__main__":
    main()
