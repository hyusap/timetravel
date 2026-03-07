# Timetravel RL Plan

## High-Level Idea

Build a temporal-control RL environment where an agent can rewind and replan from an earlier point in its own trajectory, then continue from the new timeline. In this phase we intentionally avoid parallel branching and focus on a single active timeline with explicit rewind control and meta-trajectory logging.

Core research question:

- Can a learned temporal policy decide when to rewind and how to re-instruct itself such that success under a fixed compute budget improves over strictly linear execution?

Design principles:

- Temporal control is learned compute allocation, not free search.
- World state and agent context must remain aligned at rewind points.
- Meta-trajectory data must be first-class so we can later upgrade to DAG/parallel branching.

## Product Direction (Phased)

### Phase 0 (Now): Reverse-Only Single Timeline

- Keep the primitive name `branch`, but only allow one instruction string.
- Use `branch(instruction: str, ago: int)` where `ago > 0` is required.
- No list of instructions and no sibling timelines.
- On branch, the current timeline is stopped (paused/abandoned) and execution continues from the forked rewind trajectory.
- Include `abandon()` (or equivalent pause/terminate control) in API and metadata.

### Phase 1: Learn Temporal Policy

- Train a meta-policy that chooses among: continue, branch(ago, instruction), abandon/pause.
- Optimize success under fixed budget.
- Compare against linear and hand-crafted rewind heuristics.

### Phase 2: Parallel Extension

- Upgrade from single active timeline to DAG with multiple active branches.
- Re-enable list-based `branch(instructions: List[str], ago: int)`.
- Add explicit collapse/winner-selection operations.

## Immediate Short-Term Plan (Implementation Spec)

### 1) Primitive Semantics (lock now)

- `branch(instruction: str, ago: int)`
  - `ago` must be strictly greater than 0.
  - Invalid if `ago > current_step_count`.
  - Rewind to step `current_step_count - ago`.
  - Stop current timeline at branch event (mark paused/abandoned in metadata).
  - Continue execution from the newly forked trajectory only.
  - Truncate inaccessible future from active history (or mark as inactive archive).

- `abandon()`
  - Explicitly ends or pauses the current active trajectory.
  - Must be represented in metadata/event log as a temporal control action.

### 2) Data Model Changes

- Extend action schema to support temporal control actions:
  - `kind`: `step` | `branch` | `abandon` (optionally `pause` alias).
  - `message` for normal step.
  - `instruction` + `ago` for branch.
- Extend observation/metadata with:
  - `current_step`
  - `active_timeline_id`
  - `last_branch_event`
  - `timeline_status` (`active`, `paused`, `abandoned`, `done`)

### 3) Environment State Machine

- Add timeline history storage and active pointer.
- Implement branch handler with strict `ago > 0` validation.
- Ensure branch performs stop-and-continue semantics:
  - old timeline paused/abandoned
  - new rewound trajectory becomes sole active path
- Implement abandon/pause handler.
- Reset clears all temporal state.

### 4) Meta-Trajectory Logging (required)

Record every action/event as structured logs, including:

- `event_id`, `event_type`, `step_index`, `parent_event_id`
- `timeline_id`, `source_timeline_id` (for branch)
- `ago`, `instruction`, `message`
- `reward`, `done`, `status_after_event`
- optional verifier/confidence fields for future RL features

This creates the meta-trajectory dataset now, while keeping migration path to full branching later.

### 5) Client + API Surface

- Add client helpers:
  - `branch(instruction: str, ago: int)`
  - `abandon()`
- Ensure parsing handles temporal metadata in results.
- Keep existing step behavior available for backward compatibility.

### 6) Testing Plan

Add tests for:

- branch with `ago > 0` works and rewinds correctly
- branch with `ago == 0` is rejected
- branch with `ago > step_count` is rejected
- branch stops old timeline and continues only from forked trajectory
- abandon/pause updates status correctly
- reset clears timeline and metadata

### 7) Evaluation Milestones

- M1: Temporal primitives functionally correct with tests passing.
- M2: Meta-trajectory logs exportable and analyzable.
- M3: Simple heuristic policy (rewind on failure signatures) baseline.
- M4: RL meta-policy prototype under fixed compute budget.

## Open Decisions to Resolve During Build

- Should `abandon()` terminate the episode (`done=True`) or only pause timeline state?
- Should inactive future after branch be physically truncated or archived for analysis?
- Should `pause()` be explicit or treated as an alias of `abandon()` in v0?

Default for v0 unless changed:

- `abandon()` marks trajectory abandoned and ends current episode path.
- branched-away future is archived (not active), so analysis remains possible.
- `pause()` is an alias mapped to `abandon()` behavior for now.
