---
title: Timetravel Environment Server
emoji: 🎵
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Timetravel Environment

A reverse-only temporal control environment for testing rewind-aware agent policies. The environment supports normal step actions plus temporal controls (`branch`, `abandon`/`pause`) and logs meta-trajectory data for RL.

## Quick Start

The simplest way to use the Timetravel environment is through the `TimetravelEnv` class:

```python
from timetravel import TimetravelAction, TimetravelEnv

try:
    # Create environment from Docker image
    timetravelenv = TimetravelEnv.from_docker_image("timetravel-env:latest")

    # Reset
    result = timetravelenv.reset()
    print(f"Reset: {result.observation.echoed_message}")

    # Send multiple messages
    messages = ["Hello, World!", "Testing echo", "Final message"]

    for msg in messages:
        result = timetravelenv.step(TimetravelAction(message=msg))
        print(f"Sent: '{msg}'")
        print(f"  → Echoed: '{result.observation.echoed_message}'")
        print(f"  → Length: {result.observation.message_length}")
        print(f"  → Reward: {result.reward}")

finally:
    # Always clean up
    timetravelenv.close()
```

That's it! The `TimetravelEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t timetravel-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action
**TimetravelAction** supports normal step actions and temporal controls:
- `kind` (`step` | `branch` | `abandon` | `pause`) - Action type (defaults to `step`)
- `message` (str) - Message for `step`
- `instruction` (str) - Replan instruction for `branch`
- `ago` (int) - Number of steps to rewind for `branch`, must satisfy `ago > 0`

`branch(instruction, ago)` semantics in v0:
- rewinds active timeline by `ago` steps
- stops old timeline (marked paused)
- continues only from the new forked trajectory
- no parallel sibling timelines yet

### Observation
**TimetravelObservation** contains response fields and temporal metadata:
- `echoed_message` (str) - The message echoed back
- `message_length` (int) - Length of the message
- `current_step` (int) - Active timeline step count
- `active_timeline_id` (str) - ID of active timeline
- `timeline_status` (`active` | `paused` | `abandoned` | `done`)
- `last_branch_event` (dict | null) - Most recent branch metadata
- `event_log_size` (int) - Count of structured meta-trajectory events
- `reward` (float), `done` (bool), `metadata` (dict)

### Reward
For `step` actions, reward is calculated as: `message_length × 0.1`
- "Hi" → reward: 0.2
- "Hello, World!" → reward: 1.3
- Empty message → reward: 0.0

Temporal control actions (`branch`, `abandon`, `pause`) currently return reward `0.0`.

### Temporal Control Example

```python
from timetravel import TimetravelAction, TimetravelEnv

with TimetravelEnv(base_url="http://localhost:8000") as env:
    env.reset()
    env.step(TimetravelAction(message="try approach A"))
    env.step(TimetravelAction(message="still stuck"))

    # Rewind one step and continue from a forked timeline
    env.step(TimetravelAction(kind="branch", instruction="Try approach B", ago=1))
    env.step(TimetravelAction(message="approach B attempt"))

    # Explicitly end the active timeline
    env.step(TimetravelAction(kind="abandon"))
```

## Advanced Usage

### Connecting to an Existing Server

If you already have a Timetravel environment server running, you can connect directly:

```python
from timetravel import TimetravelEnv

# Connect to existing server
timetravelenv = TimetravelEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = timetravelenv.reset()
result = timetravelenv.step(TimetravelAction(message="Hello!"))
```

Note: When connecting to an existing server, `timetravelenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from timetravel import TimetravelAction, TimetravelEnv

# Connect with context manager (auto-connects and closes)
with TimetravelEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(TimetravelAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    TimetravelEnvironment,  # Pass class, not instance
    TimetravelAction,
    TimetravelObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from timetravel import TimetravelAction, TimetravelEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with TimetravelEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(TimetravelAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Reverse-Only Benchmark (Code Door)

The repository includes a training benchmark focused on "sample from the future and send backwards" behavior using reverse-only branching.

- Environment: `benchmarks/reverse_code_door.py`
- Key constraint: success under tight budget is only achievable with `branch(instruction, ago>0)`
- Outputs: per-event meta-trajectory logs suitable for offline RL / imitation

Run the benchmark suite:

```bash
uv run python benchmarks/reverse_code_door.py
```

Export rollout data for training:

```python
from benchmarks.reverse_code_door import export_training_rollouts, rewind_policy

export_training_rollouts(
    rewind_policy,
    "data/reverse_code_door_rewind.jsonl",
    episodes=500,
    seed=42,
)
```

Shortest-path terminal reward is the default. You can tune the shape with config:

```python
from benchmarks.reverse_code_door import EpisodeConfig, evaluate_policy, rewind_policy

cfg = EpisodeConfig(
    budget=6,
    step_cost=0.0,
    success_reward=1.0,
    optimal_final_path_length=2,
    final_path_penalty=0.2,
)

metrics = evaluate_policy(rewind_policy, episodes=200, seed=0, config=cfg)
```

Success reward decays with the final active timeline path length, which directly incentivizes concise rewound solutions.

### TextWorld Temporal Wrapper (Replay-Based Rewind)

The repository also includes a TextWorld-compatible temporal wrapper that applies the same reverse-only controls:

- Module: `benchmarks/textworld_temporal.py`
- Actions: `step(command)`, `branch(instruction, ago>0)`, `abandon/pause`
- Rewind implementation: reset-and-replay from active timeline history (no native snapshot requirement)

Use it with a native TextWorld game file:

```python
from benchmarks.textworld_temporal import (
    TextWorldEpisodeConfig,
    TextWorldTemporalAction,
    make_native_textworld_env,
)

env = make_native_textworld_env(
    "games/my_game.z8",
    config=TextWorldEpisodeConfig(budget=64, step_cost=-0.01),
)

obs = env.reset(seed=0)
obs = env.step(TextWorldTemporalAction(command="look"))
obs = env.step(TextWorldTemporalAction(command="go east"))
obs = env.step(TextWorldTemporalAction(kind="branch", instruction="Try key route", ago=1))
obs = env.step(TextWorldTemporalAction(kind="abandon"))
```

If `textworld` is not installed, native backend construction raises an informative `ImportError`.

Generate worlds programmatically for curriculum runs:

```python
from benchmarks.textworld_temporal import TextWorldGenerationConfig, create_textworld_worlds

worlds = create_textworld_worlds(
    prefix="curriculum",
    count=5,
    generation=TextWorldGenerationConfig(
        output_dir="games",
        world_size=6,
        nb_objects=8,
        quest_length=4,
        theme="house",
        seed=100,
    ),
)
```

### Script-Based Training (No Notebook Required)

Notebook logic has been extracted into reusable scripts under `train/`:

- `train/train_reverse_code_door.py` - GRPO-style training loop
- `train/eval_base.py` - deterministic base/adaptor evaluation
- `train/play.py` - manual interactive environment play
- `train/reverse_code_door_agent.py` - shared prompt + action parsing helpers

Install training dependencies (once on your training machine):

```bash
uv add -r train/requirements.txt
```

Run training:

```bash
uv run python train/train_reverse_code_door.py \
  --output-dir runs/reverse_code_door \
  --num-train-steps 300 \
  --episodes-per-step 4 \
  --num-generations 4
```

Run evaluation:

```bash
uv run python train/eval_base.py --episodes 50 --seed-start 0
```

Outputs:
- Training metrics are written to `runs/.../metrics.jsonl`
- Adapter checkpoints are saved at `runs/.../checkpoint_step_*` and `runs/.../final_adapter`

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/timetravel_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
timetravel/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # TimetravelEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── timetravel_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
