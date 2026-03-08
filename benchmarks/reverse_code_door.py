from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


CODE_PATTERN = re.compile(r"\b(\d{3})\b")


@dataclass
class TemporalAction:
    kind: str = "step"
    command: str = "wait"
    instruction: str = ""
    ago: Optional[int] = None
    unlock_code: Optional[str] = None


@dataclass
class EpisodeConfig:
    budget: int = 6
    step_cost: float = -0.01
    success_reward: float = 1.0
    # Applied on incorrect unlock attempts; keep this large so brute-force is strongly disfavored.
    failure_penalty: float = -5.0
    optimal_final_path_length: int = 2
    final_path_penalty: float = 0.15
    # Penalize unlock attempts made too early in the episode (1-indexed step threshold).
    early_unlock_turn_threshold: int = 2
    early_unlock_penalty: float = -0.25
    # Penalize repeated attempts of the same code; multiplied by repeat index.
    repeated_code_penalty: float = -0.5
    # Keep default benchmark semantics unless explicitly overridden.
    end_episode_on_wrong_unlock: bool = True


class ReverseCodeDoorEnv:
    """Reverse-only benchmark where future information helps earlier choices.

    Layout is a 1D corridor:
    - start at position 0
    - door at position 1
    - oracle at position 3

    The oracle reveals a hidden 3-digit code. The episode succeeds only when
    the agent unlocks the door at position 1 with the correct code.

    Under a tight budget (default 6), linear execution cannot both scout the
    oracle and return to unlock. Rewinding with branch can make it feasible.
    """

    def __init__(self, config: Optional[EpisodeConfig] = None) -> None:
        self.config = config or EpisodeConfig()
        self._rng = random.Random(0)
        self._timelines: Dict[str, Dict[str, Any]] = {}
        self._active_timeline_id = ""
        self._timeline_counter = 0
        self._event_counter = 0
        self._total_cost = 0
        self._secret_code = ""
        self._episode_done = False
        self._last_branch_event: Optional[Dict[str, Any]] = None
        self._meta_events: List[Dict[str, Any]] = []
        self._unlock_attempt_counts: Dict[str, int] = {}
        self.reset(seed=0)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self._rng.seed(seed)

        self._timeline_counter = 1
        self._event_counter = 0
        self._total_cost = 0
        self._episode_done = False
        self._secret_code = f"{self._rng.randint(0, 999):03d}"
        self._active_timeline_id = f"timeline-{self._timeline_counter}"
        self._last_branch_event = None
        self._meta_events = []
        self._unlock_attempt_counts = {}
        self._timelines = {
            self._active_timeline_id: {
                "timeline_id": self._active_timeline_id,
                "source_timeline_id": None,
                "status": "active",
                "steps": [
                    {
                        "position": 0,
                        "visible_code": None,
                        "instruction_hint": "",
                    }
                ],
                "archived_future": [],
            }
        }
        return self._observation()

    def step(self, action: TemporalAction) -> Dict[str, Any]:
        if self._episode_done:
            return self._observation(reward=0.0, done=True, info={"event_type": "terminal_noop"})

        self._total_cost += 1
        reward = self.config.step_cost
        done = False

        if action.kind == "branch":
            reward, done, info = self._handle_branch(action)
        elif action.kind in {"abandon", "pause"}:
            reward, done, info = self._handle_abandon(action)
        else:
            reward, done, info = self._handle_step(action)

        if self.remaining_budget <= 0 and not done:
            self._episode_done = True
            done = True
            info["budget_exhausted"] = True

        return self._observation(reward=reward, done=done, info=info)

    @property
    def remaining_budget(self) -> int:
        return max(self.config.budget - self._total_cost, 0)

    @property
    def meta_events(self) -> List[Dict[str, Any]]:
        return list(self._meta_events)

    def export_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for event in self._meta_events:
                f.write(json.dumps(event) + "\n")

    def _active_timeline(self) -> Dict[str, Any]:
        return self._timelines[self._active_timeline_id]

    def _current_state(self) -> Dict[str, Any]:
        return self._active_timeline()["steps"][-1]

    def _current_step(self) -> int:
        return len(self._active_timeline()["steps"]) - 1

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"event-{self._event_counter}"

    def _log_event(
        self,
        *,
        event_type: str,
        reward: float,
        done: bool,
        message: str = "",
        instruction: str = "",
        ago: Optional[int] = None,
        source_timeline_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        event = {
            "event_id": self._next_event_id(),
            "event_type": event_type,
            "step_index": self._current_step(),
            "timeline_id": self._active_timeline_id,
            "source_timeline_id": source_timeline_id,
            "ago": ago,
            "instruction": instruction,
            "message": message,
            "reward": reward,
            "done": done,
            "status_after_event": self._active_timeline()["status"],
            "remaining_budget": self.remaining_budget,
        }
        self._meta_events.append(event)
        return event

    def _extract_code(self, text: str) -> Optional[str]:
        match = CODE_PATTERN.search(text)
        return match.group(1) if match else None

    def _handle_step(self, action: TemporalAction) -> tuple[float, bool, Dict[str, Any]]:
        state = dict(self._current_state())
        position = state["position"]
        command = action.command
        done = False
        reward = self.config.step_cost

        if command == "forward":
            position = min(position + 1, 3)
        elif command == "backward":
            position = max(position - 1, 0)
        elif command == "inspect":
            if position == 3:
                state["visible_code"] = self._secret_code
        elif command == "unlock":
            attempt = action.unlock_code or self._extract_code(state.get("instruction_hint", ""))
            reward += self._unlock_shaping_penalty(attempt=attempt)
            if position == 1 and attempt == self._secret_code:
                done = True
                self._episode_done = True
                reward += self._success_reward(final_path_length=self._current_step() + 1)
            else:
                reward += self.config.failure_penalty
                if self.config.end_episode_on_wrong_unlock:
                    done = True
                    self._episode_done = True

        next_state = {
            "position": position,
            "visible_code": state.get("visible_code"),
            "instruction_hint": state.get("instruction_hint", ""),
        }
        self._active_timeline()["steps"].append(next_state)

        event = self._log_event(
            event_type="step",
            reward=reward,
            done=done,
            message=f"{command}:{action.unlock_code or ''}",
        )
        return reward, done, {"event_id": event["event_id"], "event_type": "step", "command": command}

    def _success_reward(self, *, final_path_length: int) -> float:
        """Compute success reward from final active timeline path length."""
        extra_steps = max(final_path_length - self.config.optimal_final_path_length, 0)
        reward = self.config.success_reward - (self.config.final_path_penalty * extra_steps)
        return max(reward, 0.0)

    def _handle_branch(self, action: TemporalAction) -> tuple[float, bool, Dict[str, Any]]:
        if action.ago is None:
            raise ValueError("branch action requires ago")
        if action.ago <= 0:
            raise ValueError("branch requires ago > 0")
        if action.ago > self._current_step():
            action.ago = self._current_step()

        old_timeline_id = self._active_timeline_id
        old_timeline = self._active_timeline()
        source_step = self._current_step()
        rewind_to_step = source_step - action.ago

        old_timeline["status"] = "paused"
        old_timeline["archived_future"].append(
            {
                "from_step": rewind_to_step,
                "events": old_timeline["steps"][rewind_to_step + 1 :],
            }
        )

        self._timeline_counter += 1
        new_timeline_id = f"timeline-{self._timeline_counter}"
        new_steps = [dict(step) for step in old_timeline["steps"][: rewind_to_step + 1]]
        new_steps[-1]["instruction_hint"] = action.instruction
        self._timelines[new_timeline_id] = {
            "timeline_id": new_timeline_id,
            "source_timeline_id": old_timeline_id,
            "status": "active",
            "steps": new_steps,
            "archived_future": [],
        }
        self._active_timeline_id = new_timeline_id

        event = self._log_event(
            event_type="branch",
            reward=self.config.step_cost,
            done=False,
            instruction=action.instruction,
            ago=action.ago,
            source_timeline_id=old_timeline_id,
        )
        self._last_branch_event = {
            "event_id": event["event_id"],
            "from_timeline_id": old_timeline_id,
            "to_timeline_id": new_timeline_id,
            "source_step": source_step,
            "rewind_to_step": rewind_to_step,
            "ago": action.ago,
            "instruction": action.instruction,
        }
        return self.config.step_cost, False, {
            "event_id": event["event_id"],
            "event_type": "branch",
            "last_branch_event": self._last_branch_event,
        }

    def _handle_abandon(self, action: TemporalAction) -> tuple[float, bool, Dict[str, Any]]:
        self._active_timeline()["status"] = "abandoned"
        self._episode_done = True
        event = self._log_event(
            event_type=action.kind,
            reward=self.config.step_cost,
            done=True,
        )
        return self.config.step_cost, True, {"event_id": event["event_id"], "event_type": action.kind}

    def _unlock_shaping_penalty(self, *, attempt: Optional[str]) -> float:
        penalty = 0.0
        next_step_index = self._current_step() + 1
        if (
            self.config.early_unlock_penalty != 0.0
            and self.config.early_unlock_turn_threshold > 0
            and next_step_index <= self.config.early_unlock_turn_threshold
        ):
            penalty += self.config.early_unlock_penalty

        if attempt:
            seen = self._unlock_attempt_counts.get(attempt, 0)
            if seen > 0 and self.config.repeated_code_penalty != 0.0:
                penalty += self.config.repeated_code_penalty * seen
            self._unlock_attempt_counts[attempt] = seen + 1

        return penalty

    def _observation(
        self,
        *,
        reward: float = 0.0,
        done: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = self._current_state()
        return {
            "position": state["position"],
            "at_door": state["position"] == 1,
            "at_oracle": state["position"] == 3,
            "visible_code": state["visible_code"],
            "instruction_hint": state["instruction_hint"],
            "reward": reward,
            "done": done,
            "remaining_budget": self.remaining_budget,
            "current_step": self._current_step(),
            "active_timeline_id": self._active_timeline_id,
            "timeline_status": self._active_timeline()["status"],
            "last_branch_event": self._last_branch_event,
            "event_log_size": len(self._meta_events),
            "info": info or {},
        }


PolicyFn = Callable[[Dict[str, Any], List[Dict[str, Any]]], TemporalAction]


def linear_policy(obs: Dict[str, Any], _: List[Dict[str, Any]]) -> TemporalAction:
    if obs["done"]:
        return TemporalAction(kind="abandon")
    if obs["at_oracle"] and obs["visible_code"] is None:
        return TemporalAction(command="inspect")
    if obs["visible_code"] and not obs["at_door"]:
        return TemporalAction(command="backward")
    if obs["at_door"] and obs["visible_code"]:
        return TemporalAction(command="unlock", unlock_code=obs["visible_code"])
    return TemporalAction(command="forward")


def rewind_policy(obs: Dict[str, Any], history: List[Dict[str, Any]]) -> TemporalAction:
    if obs["done"]:
        return TemporalAction(kind="abandon")
    if obs["at_oracle"] and obs["visible_code"] is None:
        return TemporalAction(command="inspect")

    if obs["visible_code"] and obs["last_branch_event"] is None:
        for idx in range(len(history) - 1, -1, -1):
            if history[idx]["position"] == 1:
                ago = obs["current_step"] - idx
                return TemporalAction(
                    kind="branch",
                    instruction=f"Use code {obs['visible_code']} at the door",
                    ago=ago,
                )

    hint = obs["instruction_hint"]
    hinted_code = CODE_PATTERN.search(hint)
    if obs["at_door"] and hinted_code:
        return TemporalAction(command="unlock", unlock_code=hinted_code.group(1))

    if obs["at_door"] and obs["visible_code"]:
        return TemporalAction(command="unlock", unlock_code=obs["visible_code"])

    if obs["visible_code"] is None and hinted_code is None:
        if obs["at_oracle"]:
            return TemporalAction(command="inspect")
        return TemporalAction(command="forward")

    if obs["position"] < 1:
        return TemporalAction(command="forward")
    if obs["position"] > 1:
        return TemporalAction(command="backward")
    return TemporalAction(command="wait")


def random_policy(obs: Dict[str, Any], _: List[Dict[str, Any]]) -> TemporalAction:
    if obs["done"]:
        return TemporalAction(kind="abandon")
    actions = [
        TemporalAction(command="forward"),
        TemporalAction(command="backward"),
        TemporalAction(command="inspect"),
        TemporalAction(command="unlock", unlock_code=f"{random.randint(0, 999):03d}"),
    ]
    return random.choice(actions)


def rollout_episode(
    env: ReverseCodeDoorEnv,
    policy: PolicyFn,
    *,
    seed: int,
    max_steps: int = 64,
) -> Dict[str, Any]:
    obs = env.reset(seed=seed)
    history = [obs]
    total_reward = 0.0

    for _ in range(max_steps):
        action = policy(obs, history)
        obs = env.step(action)
        history.append(obs)
        total_reward += obs["reward"]
        if obs["done"]:
            break

    success = bool(obs["info"].get("command") == "unlock" and obs["reward"] > 0)
    return {
        "success": success,
        "total_reward": total_reward,
        "steps": len(history) - 1,
        "remaining_budget": obs["remaining_budget"],
        "event_log_size": obs["event_log_size"],
        "used_branch": any(e["event_type"] == "branch" for e in env.meta_events),
        "meta_events": env.meta_events,
    }


def evaluate_policy(
    policy: PolicyFn,
    *,
    episodes: int = 200,
    seed: int = 0,
    config: Optional[EpisodeConfig] = None,
) -> Dict[str, Any]:
    env = ReverseCodeDoorEnv(config=config)
    successes = 0
    reward_sum = 0.0
    branch_count = 0

    for i in range(episodes):
        result = rollout_episode(env, policy, seed=seed + i)
        successes += int(result["success"])
        reward_sum += result["total_reward"]
        branch_count += int(result["used_branch"])

    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "avg_reward": reward_sum / episodes,
        "branch_rate": branch_count / episodes,
        "budget": (config or EpisodeConfig()).budget,
    }


def export_training_rollouts(
    policy: PolicyFn,
    output_path: str,
    *,
    episodes: int = 100,
    seed: int = 0,
    config: Optional[EpisodeConfig] = None,
) -> None:
    env = ReverseCodeDoorEnv(config=config)
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(episodes):
            result = rollout_episode(env, policy, seed=seed + i)
            for event in result["meta_events"]:
                row = {
                    "episode_index": i,
                    "success": result["success"],
                    "budget": env.config.budget,
                    **event,
                }
                f.write(json.dumps(row) + "\n")


def benchmark_suite(episodes: int = 200, seed: int = 0) -> Dict[str, Dict[str, Any]]:
    return {
        "linear": evaluate_policy(linear_policy, episodes=episodes, seed=seed),
        "rewind": evaluate_policy(rewind_policy, episodes=episodes, seed=seed),
        "random": evaluate_policy(random_policy, episodes=episodes, seed=seed),
    }


if __name__ == "__main__":
    results = benchmark_suite(episodes=100)
    print(json.dumps(results, indent=2))
