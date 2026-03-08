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
from train.reverse_code_door_agent import SYSTEM_PROMPT, format_action, infer_success, obs_to_text, parse_action


def _generate_until_action(
    model,
    tokenizer,
    prompt_ids,
    *,
    max_total_new_tokens: int,
    chunk_new_tokens: int,
    temperature: float,
    do_sample: bool,
):
    """Generate incrementally until a valid ACTION line is emitted or token cap is reached."""
    import torch

    generated = torch.empty(0, dtype=prompt_ids.dtype, device=prompt_ids.device)
    cursor = prompt_ids
    tokens_left = max_total_new_tokens

    while tokens_left > 0:
        step_tokens = min(chunk_new_tokens, tokens_left)
        attention_mask = torch.ones_like(cursor, device=cursor.device)
        out = model.generate(
            cursor,
            attention_mask=attention_mask,
            max_new_tokens=step_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_ids = out[0][cursor.shape[1] :]
        if len(new_ids) == 0:
            break

        generated = torch.cat([generated, new_ids], dim=0)
        cursor = torch.cat([cursor[0], new_ids]).unsqueeze(0)
        tokens_left -= len(new_ids)

        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if "action:" in text.lower() and parse_action(text) is not None:
            break

    return generated


def collect_episode(
    model,
    tokenizer,
    *,
    seed: int,
    max_episode_steps: int,
    generation_max_new_tokens: int,
    temperature: float,
    debug_prefix: str | None = None,
    debug_full_tokens: bool = False,
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
            prompt_messages = messages + [{"role": "assistant", "content": "ACTION:"}]
            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=False,
                return_tensors="pt",
            ).to(model.device)

            action_ids = _generate_until_action(
                model,
                tokenizer,
                prompt_ids,
                max_total_new_tokens=generation_max_new_tokens,
                chunk_new_tokens=min(32, generation_max_new_tokens),
                temperature=temperature,
                do_sample=True,
            )
            action_text = tokenizer.decode(action_ids, skip_special_tokens=True).strip()
            action = parse_action(action_text)
            if action is None:
                from benchmarks.reverse_code_door import TemporalAction

                action = TemporalAction(command="wait")

            obs = env.step(action)
            if debug_prefix is not None:
                action_kind = action.kind if action.kind != "step" else action.command
                print(
                    f"{debug_prefix} step={step+1} action={action_kind!r} raw={action_text!r} "
                    f"pos={obs['position']} reward={obs['reward']:.3f} done={obs['done']}",
                    flush=True,
                )
                if debug_full_tokens:
                    full_decoded = tokenizer.decode(
                        action_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    print(
                        f"{debug_prefix} tokens={len(action_ids)} token_ids={action_ids.tolist()}",
                        flush=True,
                    )
                    print(
                        f"{debug_prefix} full_decoded={full_decoded!r}",
                        flush=True,
                    )
            messages.append({"role": "assistant", "content": format_action(action)})
            transitions.append((prompt_ids[0].cpu(), action_ids.cpu(), float(obs["reward"])))

            if obs["done"]:
                break

    model.train()
    return transitions, infer_success(obs)


def compute_episode_return(transitions: Iterable[tuple[torch.Tensor, torch.Tensor, float]]) -> float:
    return sum(reward for _, _, reward in transitions)


def policy_loss(model, prompt_ids: torch.Tensor, action_ids: torch.Tensor, advantage: float) -> torch.Tensor:
    """Compute mean token NLL over action tokens weighted by advantage."""
    import torch

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
                prompt_messages = messages + [{"role": "assistant", "content": "ACTION:"}]
                prompt_ids = tokenizer.apply_chat_template(
                    prompt_messages,
                    add_generation_prompt=False,
                    return_tensors="pt",
                ).to(model.device)
                action_ids = _generate_until_action(
                    model,
                    tokenizer,
                    prompt_ids,
                    max_total_new_tokens=max_new_tokens,
                    chunk_new_tokens=min(32, max_new_tokens),
                    temperature=0.0,
                    do_sample=False,
                )
                action_text = tokenizer.decode(action_ids, skip_special_tokens=True).strip()
                action = parse_action(action_text)
                if action is None:
                    from benchmarks.reverse_code_door import TemporalAction

                    action = TemporalAction(command="wait")
                messages.append({"role": "assistant", "content": format_action(action)})
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
    parser = argparse.ArgumentParser(description="Train an Unsloth model on Reverse Code Door (script version)")
    parser.add_argument("--model-name", default="unsloth/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", default="runs/reverse_code_door")

    parser.add_argument("--max-seq-length", type=int, default=2048)
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
    parser.add_argument("--print-actions", action="store_true")
    parser.add_argument("--print-actions-train-steps", type=int, default=3)
    parser.add_argument(
        "--print-full-tokens",
        action="store_true",
        help="When printing actions, also print full generated token IDs and full decoded text.",
    )
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
            for seed_idx, seed in enumerate(seeds):
                group_rollouts = []
                for gen_idx in range(args.num_generations):
                    debug_prefix = None
                    if args.print_actions and train_step < args.print_actions_train_steps and seed_idx == 0 and gen_idx == 0:
                        debug_prefix = f"[train_step={train_step} seed={seed}]"
                    transitions, success = collect_episode(
                        model,
                        tokenizer,
                        seed=seed,
                        max_episode_steps=args.max_episode_steps,
                        generation_max_new_tokens=args.generation_max_new_tokens,
                        temperature=args.temperature,
                        debug_prefix=debug_prefix,
                        debug_full_tokens=args.print_full_tokens,
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
