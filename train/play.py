"""
Interactive CLI to play the Reverse Code Door environment.
Run: python train/play.py [--seed N] [--budget N]
"""
import sys
import argparse

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from benchmarks.reverse_code_door import ReverseCodeDoorEnv, TemporalAction, EpisodeConfig

HELP = """
Commands:
  f / forward          move forward
  b / backward         move backward
  i / inspect          inspect oracle (reveals code at pos 3)
  u <code> / unlock    unlock door with 3-digit code
  branch <ago> [msg]   rewind <ago> steps, attach optional instruction
  abandon              give up
  r / reset            start a new episode
  q / quit             exit
  ? / help             show this message
"""

POSITIONS = {0: "start", 1: "door", 2: "corridor", 3: "oracle"}


def render(obs: dict, secret: str):
    corridor = []
    for p in range(4):
        label = f"[{p}:{POSITIONS[p]}]"
        if p == obs["position"]:
            label = f">>>{label}<<<"
        corridor.append(label)
    print("\n" + " -- ".join(corridor))
    print(f"Budget: {obs['remaining_budget']}  Step: {obs['current_step']}  Timeline: {obs['active_timeline_id']}")
    if obs["visible_code"]:
        print(f"Visible code: {obs['visible_code']}")
    if obs["instruction_hint"]:
        print(f"Hint: {obs['instruction_hint']}")
    if obs["last_branch_event"]:
        e = obs["last_branch_event"]
        print(f"Last branch: rewound {e['ago']} step(s) from {e['from_timeline_id']} → {e['to_timeline_id']}")
    if obs["reward"] != 0:
        print(f"Reward: {obs['reward']:+.3f}")


def parse_input(line: str):
    parts = line.strip().split(None, 2)
    if not parts:
        return None
    cmd = parts[0].lower()

    if cmd in ("f", "forward"):
        return TemporalAction(command="forward")
    if cmd in ("b", "backward"):
        return TemporalAction(command="backward")
    if cmd in ("i", "inspect"):
        return TemporalAction(command="inspect")
    if cmd in ("u", "unlock"):
        code = parts[1] if len(parts) > 1 else ""
        return TemporalAction(command="unlock", unlock_code=code)
    if cmd == "branch":
        if len(parts) < 2:
            print("Usage: branch <ago> [instruction]")
            return None
        try:
            ago = int(parts[1])
        except ValueError:
            print("branch: <ago> must be an integer")
            return None
        instruction = parts[2] if len(parts) > 2 else ""
        return TemporalAction(kind="branch", ago=ago, instruction=instruction)
    if cmd in ("abandon",):
        return TemporalAction(kind="abandon")
    return cmd  # pass through control commands (reset, quit, help)


def play(seed: int, budget: int):
    config = EpisodeConfig(budget=budget)
    env = ReverseCodeDoorEnv(config=config)
    obs = env.reset(seed=seed)

    print(f"\nReverse Code Door  |  seed={seed}  budget={budget}")
    print("(secret code is hidden — visit the oracle at pos 3 to reveal it)")
    print(HELP)
    render(obs, env._secret_code)

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue

        if line.lower() in ("?", "help"):
            print(HELP)
            continue

        if line.lower() in ("q", "quit", "exit"):
            print("Bye.")
            break

        if line.lower() in ("r", "reset"):
            seed += 1
            obs = env.reset(seed=seed)
            print(f"\nReset — new episode (seed={seed})")
            render(obs, env._secret_code)
            continue

        result = parse_input(line)
        if result is None:
            continue
        if isinstance(result, str):
            print(f"Unknown command: {result!r}  (type ? for help)")
            continue

        action = result
        try:
            obs = env.step(action)
        except ValueError as e:
            print(f"Invalid action: {e}")
            continue

        render(obs, env._secret_code)

        if obs["done"]:
            success = obs["info"].get("command") == "unlock" and obs["reward"] > 0
            if success:
                print("\n✓ Door unlocked! You win.")
            else:
                print(f"\n✗ Episode over. Secret was: {env._secret_code}")
            print("Type 'r' to play again or 'q' to quit.")


def main():
    parser = argparse.ArgumentParser(description="Play Reverse Code Door interactively")
    parser.add_argument("--seed", type=int, default=0, help="Episode seed")
    parser.add_argument("--budget", type=int, default=6, help="Step budget")
    args = parser.parse_args()
    play(args.seed, args.budget)


if __name__ == "__main__":
    main()
