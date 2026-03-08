"""Standalone GRPO-style trainer for Reverse Code Door without notebooks.

Example:
  uv run python train/train_reverse_code_door.py \\
    --output-dir runs/reverse_code_door \\
    --num-train-steps 300 --episodes-per-step 4 --num-generations 4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.reverse_code_door import ReverseCodeDoorEnv
from train.reverse_code_door_agent import SYSTEM_PROMPT, infer_success, obs_to_text, parse_action


def collect_episode(
    model,
    tokenizer,
    *,
    seed: int,
    max_episode_steps: int,
    generation_max_new_tokens: int,
    temperature: float,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, float]], bool]:
    """Roll out one sampled episode and return per-step supervised transitions."""
    import torch

    env = ReverseCodeDoorEnv()
    obs = env.reset(seed=seed)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    transitions: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    model.eval()
    with torch.inference_mode():
        for step in range(max_episode_steps):
            messages.append({"role": "user", "content": obs_to_text(obs, step + 1)})
            prompt_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)

            out = model.generate(
                prompt_ids,
                max_new_tokens=generation_max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            action_ids = out[0][prompt_ids.shape[1] :]
            action_text = tokenizer.decode(action_ids, skip_special_tokens=True).strip()
            action = parse_action(action_text)
            if action is None:
                from benchmarks.reverse_code_door import TemporalAction

                action = TemporalAction(kind="abandon")

            obs = env.step(action)
            messages.append({"role": "assistant", "content": action_text})
            transitions.append((prompt_ids[0].cpu(), action_ids.cpu(), float(obs["reward"])))

            if obs["done"]:
                break

    model.train()
    return transitions, infer_success(obs)


def compute_episode_return(transitions: Iterable[tuple[torch.Tensor, torch.Tensor, float]]) -> float:
    return sum(reward for _, _, reward in transitions)


def policy_loss(model, prompt_ids: torch.Tensor, action_ids: torch.Tensor, advantage: float) -> torch.Tensor:
    """Compute mean token NLL over action tokens weighted by advantage."""

    input_ids = torch.cat([prompt_ids, action_ids]).unsqueeze(0).to(model.device)
    labels = torch.full_like(input_ids, -100)
    labels[0, len(prompt_ids) :] = action_ids

    outputs = model(input_ids=input_ids, labels=labels)
    return outputs.loss * advantage


def evaluate_model(model, tokenizer, *, seeds: range, max_episode_steps: int, max_new_tokens: int) -> dict:
    import torch

    env = ReverseCodeDoorEnv()
    successes = 0
    branch_used = 0

    model.eval()
    with torch.inference_mode():
        for seed in seeds:
            obs = env.reset(seed=seed)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            for step in range(max_episode_steps):
                messages.append({"role": "user", "content": obs_to_text(obs, step + 1)})
                prompt_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(model.device)
                out = model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                action_text = tokenizer.decode(out[0][prompt_ids.shape[1] :], skip_special_tokens=True).strip()
                action = parse_action(action_text)
                if action is None:
                    from benchmarks.reverse_code_door import TemporalAction

                    action = TemporalAction(kind="abandon")
                messages.append({"role": "assistant", "content": action_text})
                obs = env.step(action)
                if obs["done"]:
                    break

            successes += int(infer_success(obs))
            branch_used += int(any(e["event_type"] == "branch" for e in env.meta_events))

    episodes = len(seeds)
    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "branch_rate": branch_used / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GPT-OSS on Reverse Code Door (script version)")
    parser.add_argument("--model-name", default="unsloth/gpt-oss-20b")
    parser.add_argument("--output-dir", default="runs/reverse_code_door")

    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--lora-rank", type=int, default=4)

    parser.add_argument("--num-train-steps", type=int, default=300)
    parser.add_argument("--episodes-per-step", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--max-episode-steps", type=int, default=10)
    parser.add_argument("--generation-max-new-tokens", type=int, default=32)
    parser.add_argument("--seed-min", type=int, default=0)
    parser.add_argument("--seed-max", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=3407)

    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from torch.nn.utils import clip_grad_norm_
    from torch.optim import AdamW

    random.seed(args.rng_seed)
    torch.manual_seed(args.rng_seed)

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError("unsloth is required. Install training deps before running this script.") from exc

    print(f"Loading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        max_seq_length=args.max_seq_length,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=args.rng_seed,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metrics_file = out_dir / "metrics.jsonl"

    print("Starting training...")
    model.train()
    with metrics_file.open("w", encoding="utf-8") as mf:
        for train_step in range(args.num_train_steps):
            optimizer.zero_grad()

            step_successes = 0
            step_returns: list[float] = []
            total_loss = torch.tensor(0.0, device=model.device)
            num_transitions = 0

            seeds = [random.randint(args.seed_min, args.seed_max) for _ in range(args.episodes_per_step)]
            for seed in seeds:
                group_rollouts = []
                for _ in range(args.num_generations):
                    transitions, success = collect_episode(
                        model,
                        tokenizer,
                        seed=seed,
                        max_episode_steps=args.max_episode_steps,
                        generation_max_new_tokens=args.generation_max_new_tokens,
                        temperature=args.temperature,
                    )
                    ret = compute_episode_return(transitions)
                    group_rollouts.append((transitions, ret, success))
                    step_successes += int(success)

                returns = [r for _, r, _ in group_rollouts]
                mean_ret = sum(returns) / len(returns)
                std_ret = (sum((r - mean_ret) ** 2 for r in returns) / len(returns)) ** 0.5 + 1e-8
                advantages = [(r - mean_ret) / std_ret for r in returns]
                step_returns.extend(returns)

                for (transitions, _, _), advantage in zip(group_rollouts, advantages):
                    for prompt_ids, action_ids, _ in transitions:
                        if len(action_ids) == 0:
                            continue
                        loss = policy_loss(model, prompt_ids, action_ids, -advantage)
                        total_loss = total_loss + loss
                        num_transitions += 1

            if num_transitions > 0:
                (total_loss / num_transitions).backward()
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            denom = args.episodes_per_step * args.num_generations
            success_rate = step_successes / denom
            avg_return = sum(step_returns) / len(step_returns)
            loss_value = total_loss.item() / max(num_transitions, 1)

            row = {
                "step": train_step,
                "success_rate": success_rate,
                "avg_return": avg_return,
                "loss": loss_value,
            }

            if args.eval_every > 0 and (train_step % args.eval_every == 0):
                eval_metrics = evaluate_model(
                    model,
                    tokenizer,
                    seeds=range(2000, 2000 + args.eval_episodes),
                    max_episode_steps=args.max_episode_steps,
                    max_new_tokens=args.generation_max_new_tokens,
                )
                row.update({f"eval_{k}": v for k, v in eval_metrics.items()})

            mf.write(json.dumps(row) + "\n")
            mf.flush()

            if train_step % 10 == 0:
                print(
                    f"step {train_step:4d} | success={success_rate:.0%} | avg_return={avg_return:.3f} | loss={loss_value:.4f}",
                    flush=True,
                )

            if args.save_every > 0 and train_step > 0 and (train_step % args.save_every == 0):
                ckpt_dir = out_dir / f"checkpoint_step_{train_step}"
                model.save_pretrained(str(ckpt_dir))
                tokenizer.save_pretrained(str(ckpt_dir))

    final_dir = out_dir / "final_adapter"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Done. Saved adapter to: {final_dir}")


if __name__ == "__main__":
    main()
