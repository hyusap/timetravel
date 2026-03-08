"""Evaluate a base model as an agent on ReverseCodeDoorEnv.

Example:
  uv run python train/eval_base.py --episodes 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.reverse_code_door import ReverseCodeDoorEnv, TemporalAction
from train.reverse_code_door_agent import SYSTEM_PROMPT, format_action, infer_success, obs_to_text, parse_action


def run_episode(model, tokenizer, seed: int, max_steps: int, max_new_tokens: int, verbose: bool = False) -> dict:
    import torch

    env = ReverseCodeDoorEnv()
    obs = env.reset(seed=seed)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    model.eval()
    with torch.inference_mode():
        for step in range(max_steps):
            messages.append({"role": "user", "content": obs_to_text(obs, step + 1)})

            prompt_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            attention_mask = torch.ones_like(prompt_ids, device=prompt_ids.device)
            out = model.generate(
                prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            action_text = tokenizer.decode(out[0][prompt_ids.shape[1] :], skip_special_tokens=True).strip()
            action = parse_action(action_text) or TemporalAction(command="wait")
            messages.append({"role": "assistant", "content": format_action(action)})
            obs = env.step(action)

            if verbose:
                kind = action.kind if action.kind != "step" else action.command
                print(
                    f"  step {step+1} [{kind}] -> pos={obs['position']} "
                    f"budget={obs['remaining_budget']} reward={obs['reward']:.3f}",
                    flush=True,
                )

            if obs["done"]:
                break

    used_branch = any(e["event_type"] == "branch" for e in env.meta_events)
    return {"success": infer_success(obs), "used_branch": used_branch, "steps": len(env.meta_events)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base model on Reverse Code Door")
    parser.add_argument("--model-name", default="unsloth/Qwen3-14B-unsloth-bnb-4bit")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    import torch

    from unsloth import FastLanguageModel

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        load_in_4bit=True,
        max_seq_length=2048,
    )

    if args.verbose:
        print(f"\n=== Single episode (seed={args.seed_start}) ===")
        first = run_episode(
            model,
            tokenizer,
            seed=args.seed_start,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            verbose=True,
        )
        print(f"Result: success={first['success']} used_branch={first['used_branch']}\n")

    successes = 0
    branch_used = 0
    for offset in range(args.episodes):
        seed = args.seed_start + offset
        result = run_episode(
            model,
            tokenizer,
            seed=seed,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
        )
        successes += int(result["success"])
        branch_used += int(result["used_branch"])
        print(f"seed={seed:4d} success={result['success']} branch={result['used_branch']}", flush=True)

    print(f"\nSuccess rate: {successes}/{args.episodes} = {successes/args.episodes:.0%}")
    print(f"Branch rate : {branch_used}/{args.episodes} = {branch_used/args.episodes:.0%}")


if __name__ == "__main__":
    main()
