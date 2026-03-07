from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol


class TextWorldBackend(Protocol):
    """Minimal backend contract to support temporal wrapping and testing."""

    def reset(self, seed: Optional[int] = None) -> tuple[str, float, bool, Dict[str, Any]]:
        ...

    def step(self, command: str) -> tuple[str, float, bool, Dict[str, Any]]:
        ...


class NativeTextWorldBackend:
    """Adapter around textworld.start(gamefile) runtime."""

    def __init__(self, game_file: str) -> None:
        try:
            import textworld  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "textworld is required for NativeTextWorldBackend. Install it with `pip install textworld`."
            ) from exc

        self._env = textworld.start(game_file)

    def reset(self, seed: Optional[int] = None) -> tuple[str, float, bool, Dict[str, Any]]:
        # Native TextWorld reset does not currently consume a seed parameter.
        game_state = self._env.reset()
        feedback = getattr(game_state, "feedback", "")
        score = float(getattr(game_state, "score", 0.0) or 0.0)
        done = bool(getattr(game_state, "game_ended", False))
        return feedback, score, done, {}

    def step(self, command: str) -> tuple[str, float, bool, Dict[str, Any]]:
        game_state, score, done = self._env.step(command)
        feedback = getattr(game_state, "feedback", "")
        return feedback, float(score), bool(done), {}


@dataclass
class TextWorldTemporalAction:
    kind: str = "step"
    command: str = "look"
    instruction: str = ""
    ago: Optional[int] = None


@dataclass
class TextWorldEpisodeConfig:
    budget: int = 50
    step_cost: float = 0.0


@dataclass
class TextWorldGenerationConfig:
    output_dir: str = "tw_games"
    world_size: int = 5
    nb_objects: int = 10
    quest_length: int = 5
    theme: str = "house"
    seed: int = 0
    force: bool = True


class TemporalTextWorldEnv:
    """Reverse-only temporal wrapper for TextWorld-like environments.

    Rewind is implemented via deterministic replay from reset through preserved
    action history. This keeps semantics aligned with `branch(instruction, ago)`
    without requiring native snapshot support from the backend.
    """

    def __init__(
        self,
        backend_factory: Callable[[], TextWorldBackend],
        config: Optional[TextWorldEpisodeConfig] = None,
    ) -> None:
        self._backend_factory = backend_factory
        self.config = config or TextWorldEpisodeConfig()

        self._backend: TextWorldBackend | None = None
        self._episode_seed = 0
        self._timeline_counter = 0
        self._event_counter = 0
        self._total_cost = 0
        self._active_timeline_id = ""
        self._timelines: Dict[str, Dict[str, Any]] = {}
        self._meta_events: List[Dict[str, Any]] = []
        self._episode_done = False
        self._last_branch_event: Optional[Dict[str, Any]] = None

    def reset(self, seed: int = 0) -> Dict[str, Any]:
        self._episode_seed = seed
        self._timeline_counter = 1
        self._event_counter = 0
        self._total_cost = 0
        self._episode_done = False
        self._active_timeline_id = f"timeline-{self._timeline_counter}"
        self._meta_events = []
        self._last_branch_event = None

        self._backend = self._backend_factory()
        feedback, score, done, info = self._backend.reset(seed=seed)

        self._timelines = {
            self._active_timeline_id: {
                "timeline_id": self._active_timeline_id,
                "source_timeline_id": None,
                "status": "active",
                "instruction_hint": "",
                "actions": [],
                "states": [
                    {
                        "feedback": feedback,
                        "score": score,
                        "done": done,
                        "info": info,
                    }
                ],
                "archived_future": [],
            }
        }
        return self._observation(reward=0.0, done=done, info={"event_type": "reset"})

    def step(self, action: TextWorldTemporalAction) -> Dict[str, Any]:
        if self._episode_done:
            return self._observation(reward=0.0, done=True, info={"event_type": "terminal_noop"})

        self._total_cost += 1

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

    def _current_step(self) -> int:
        return len(self._active_timeline()["actions"])

    def _current_state(self) -> Dict[str, Any]:
        return self._active_timeline()["states"][-1]

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"event-{self._event_counter}"

    def _log_event(
        self,
        *,
        event_type: str,
        reward: float,
        done: bool,
        command: str = "",
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
            "message": command,
            "reward": reward,
            "done": done,
            "status_after_event": self._active_timeline()["status"],
            "remaining_budget": self.remaining_budget,
        }
        self._meta_events.append(event)
        return event

    def _handle_step(self, action: TextWorldTemporalAction) -> tuple[float, bool, Dict[str, Any]]:
        if self._backend is None:  # pragma: no cover
            raise RuntimeError("Environment not initialized. Call reset() first.")

        command = action.command.strip()
        if not command:
            raise ValueError("step action requires non-empty command")

        prev_score = float(self._current_state().get("score", 0.0))
        feedback, score, done, backend_info = self._backend.step(command)

        shaped_reward = (score - prev_score) + self.config.step_cost
        if done:
            self._episode_done = True
            self._active_timeline()["status"] = "done"

        self._active_timeline()["actions"].append(command)
        self._active_timeline()["states"].append(
            {
                "feedback": feedback,
                "score": score,
                "done": done,
                "info": backend_info,
            }
        )

        event = self._log_event(event_type="step", reward=shaped_reward, done=done, command=command)
        return shaped_reward, done, {"event_type": "step", "event_id": event["event_id"], "command": command}

    def _handle_branch(self, action: TextWorldTemporalAction) -> tuple[float, bool, Dict[str, Any]]:
        if action.ago is None:
            raise ValueError("branch action requires ago")
        if action.ago <= 0:
            raise ValueError("branch requires ago > 0")
        if action.ago > self._current_step():
            raise ValueError("branch ago cannot exceed current_step")

        old_timeline_id = self._active_timeline_id
        old_timeline = self._active_timeline()
        source_step = self._current_step()
        rewind_to_step = source_step - action.ago

        old_timeline["status"] = "paused"
        old_timeline["archived_future"].append(
            {
                "from_step": rewind_to_step,
                "actions": old_timeline["actions"][rewind_to_step:],
                "states": old_timeline["states"][rewind_to_step + 1 :],
            }
        )

        self._timeline_counter += 1
        new_timeline_id = f"timeline-{self._timeline_counter}"
        preserved_actions = list(old_timeline["actions"][:rewind_to_step])

        self._timelines[new_timeline_id] = {
            "timeline_id": new_timeline_id,
            "source_timeline_id": old_timeline_id,
            "status": "active",
            "instruction_hint": action.instruction,
            "actions": preserved_actions,
            "states": [],
            "archived_future": [],
        }
        self._active_timeline_id = new_timeline_id
        self._replay_active_timeline()

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
            "event_type": "branch",
            "event_id": event["event_id"],
            "last_branch_event": self._last_branch_event,
        }

    def _handle_abandon(self, action: TextWorldTemporalAction) -> tuple[float, bool, Dict[str, Any]]:
        self._active_timeline()["status"] = "abandoned"
        self._episode_done = True
        event = self._log_event(event_type=action.kind, reward=self.config.step_cost, done=True)
        return self.config.step_cost, True, {"event_type": action.kind, "event_id": event["event_id"]}

    def _replay_active_timeline(self) -> None:
        """Reset backend and replay active timeline commands to reconstruct state."""
        self._backend = self._backend_factory()
        feedback, score, done, info = self._backend.reset(seed=self._episode_seed)

        timeline = self._active_timeline()
        commands = list(timeline["actions"])
        timeline["states"] = [
            {
                "feedback": feedback,
                "score": score,
                "done": done,
                "info": info,
            }
        ]

        for command in commands:
            feedback, score, done, info = self._backend.step(command)
            timeline["states"].append(
                {
                    "feedback": feedback,
                    "score": score,
                    "done": done,
                    "info": info,
                }
            )

        self._episode_done = bool(timeline["states"][-1]["done"])
        timeline["status"] = "done" if self._episode_done else "active"

    def _observation(self, *, reward: float, done: bool, info: Dict[str, Any]) -> Dict[str, Any]:
        state = self._current_state()
        return {
            "feedback": state.get("feedback", ""),
            "score": state.get("score", 0.0),
            "instruction_hint": self._active_timeline().get("instruction_hint", ""),
            "reward": reward,
            "done": done,
            "remaining_budget": self.remaining_budget,
            "current_step": self._current_step(),
            "active_timeline_id": self._active_timeline_id,
            "timeline_status": self._active_timeline()["status"],
            "last_branch_event": self._last_branch_event,
            "event_log_size": len(self._meta_events),
            "info": info,
        }


def make_native_textworld_env(
    game_file: str,
    *,
    config: Optional[TextWorldEpisodeConfig] = None,
) -> TemporalTextWorldEnv:
    """Create a temporal TextWorld env backed by the native TextWorld runtime."""

    return TemporalTextWorldEnv(
        backend_factory=lambda: NativeTextWorldBackend(game_file),
        config=config,
    )


def create_textworld_world(
    *,
    world_name: str,
    generation: Optional[TextWorldGenerationConfig] = None,
) -> str:
    """Generate a TextWorld custom game file using `tw-make custom`."""

    cfg = generation or TextWorldGenerationConfig()
    tw_make = shutil.which("tw-make")
    if tw_make is None:
        raise ImportError("`tw-make` not found. Install TextWorld with `uv add textworld`.")

    output_path = Path(cfg.output_dir) / f"{world_name}.z8"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        tw_make,
        "custom",
        "--world-size",
        str(cfg.world_size),
        "--nb-objects",
        str(cfg.nb_objects),
        "--quest-length",
        str(cfg.quest_length),
        "--theme",
        cfg.theme,
        "--seed",
        str(cfg.seed),
        "--output",
        str(output_path),
    ]
    if cfg.force:
        cmd.append("-f")

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return str(output_path)


def create_textworld_worlds(
    *,
    prefix: str,
    count: int,
    generation: Optional[TextWorldGenerationConfig] = None,
) -> List[str]:
    """Generate a sequence of worlds with incrementing seeds."""

    if count <= 0:
        raise ValueError("count must be > 0")

    base = generation or TextWorldGenerationConfig()
    worlds: List[str] = []
    for idx in range(count):
        worlds.append(
            create_textworld_world(
                world_name=f"{prefix}_{idx}",
                generation=TextWorldGenerationConfig(
                    output_dir=base.output_dir,
                    world_size=base.world_size,
                    nb_objects=base.nb_objects,
                    quest_length=base.quest_length,
                    theme=base.theme,
                    seed=base.seed + idx,
                    force=base.force,
                ),
            )
        )

    return worlds
