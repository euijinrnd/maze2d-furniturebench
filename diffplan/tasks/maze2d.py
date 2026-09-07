"""D4RL Maze2D: dataset loading, environment construction, rollout, plotting.

Needs d4rl (and therefore mujoco-py); it must never be imported by anything that also wants
isaacgym -- see diffplan/tasks/__init__.py.
"""
from __future__ import annotations

import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import numpy as np

import gym


@contextmanager
def _quiet():
    """Swallow stdout/stderr. d4rl is chatty about mujoco bindings it does not need."""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull), redirect_stdout(fnull):
            yield


# Imported purely for the side effect of registering the maze2d-* gym ids.
with _quiet():
    import d4rl  # noqa: F401

from diffplan.diffusion import plan

# parse_maze() encodes the maze with WALL=10, EMPTY=11, GOAL=12.
WALL = 10

# Upstream's episode segmentation threshold (diffuser/datasets/preprocessing.py).
GOAL_THRESHOLD = 0.5


def make_env(env_id):
    """gym.make + unwrap, exposing .max_episode_steps, ._target and .name.

    The UNWRAPPED env (as upstream's load_environment returns) so the TimeLimit wrapper cannot
    truncate a rollout at a different horizon than the one we loop over, and so ._target /
    .state_vector() / .get_dataset() are reachable. The step budget is copied onto the raw env.
    """
    with _quiet():
        wrapped = gym.make(env_id)
    env = wrapped.unwrapped
    env.max_episode_steps = wrapped._max_episode_steps
    env.name = env_id
    return env


def load_episodes(env):
    """Split the raw d4rl maze2d dataset into per-episode dicts of observations/actions/rewards.

    Raw maze2d data is effectively ONE ~1M-step trajectory from a waypoint controller that is
    re-targeted whenever it reaches the current goal, and the shipped 'timeouts' field does not
    reflect that. Trusting it would make an "episode" span most of the dataset, so conditioning on
    (obs[0], obs[H-1]) would inpaint between two unrelated points. We re-segment the way
    maze2d_set_terminals does: cut where the agent LEAVES the goal region, because that is the
    moment the demonstrator switched goals.
    """
    dataset = env.get_dataset()
    observations, actions, rewards = dataset['observations'], dataset['actions'], dataset['rewards']

    goal = np.asarray(env._target, dtype=np.float64)
    at_goal = np.linalg.norm(observations[:, :2] - goal, axis=-1) < GOAL_THRESHOLD
    timeouts = np.zeros(len(observations), dtype=bool)
    timeouts[:-1] = at_goal[:-1] & ~at_goal[1:]  # timeout at t iff at goal at t and not at t + 1

    # Each done index is the last step of its episode. The tail after the final done is dropped,
    # as upstream's sequence_dataset() also drops it, so the training set stays the one the
    # published numbers came from.
    ends = np.flatnonzero(timeouts | dataset['terminals'].astype(bool))
    starts = np.concatenate([[0], ends[:-1] + 1])

    episodes, n_dropped = [], 0
    for s, e in zip(starts, ends):
        # `e` is inclusive, and the final transition of every episode is dropped the way
        # process_maze2d_episode does (it builds next_observations = observations[1:] and
        # truncates the rest to match). We do not need next_observations, but we do need the same
        # window count and content.
        if e - s < 1:
            n_dropped += 1
            continue
        episodes.append({
            'observations': observations[s:e].astype(np.float32),
            'actions': actions[s:e].astype(np.float32),
            'rewards': rewards[s:e].astype(np.float32),
        })

    lengths = np.array([len(ep['observations']) for ep in episodes])
    print('[ tasks/maze2d ] {}: {} episodes | min {} | max {} | mean {:.0f} | dropped {}'
          .format(getattr(env, 'name', 'maze2d'), len(episodes), lengths.min(),
                  lengths.max(), lengths.mean(), n_dropped))
    return episodes


def rollout(env, diffusion, normalizer, cfg, seed=None):
    """One goal-conditioned episode: plan ONCE, then track the plan open loop.

    THE DIFFUSED ACTIONS ARE THROWN AWAY. That is upstream's scripts/plan_maze2d.py, not a
    simplification here: it computes

        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

    i.e. a hand-written PD controller on the predicted STATE sequence with unit gains, leaving
    `samples.actions` unused. The published 113.9 / 121.5 / 123.0 are those PD numbers, so the
    model here is a trajectory planner, not a policy. Open loop is upstream's choice too: "the
    open-loop plans are good enough for maze2d that we really only need to plan once".
    """
    if seed is not None:
        env.seed(seed)
    observation = env.reset()
    target = np.asarray(env._target, dtype=np.float64)

    # Inpaint start and goal. The goal is a full observation, so we ask for zero VELOCITY at
    # arrival too -- otherwise the plan may blow through the goal at speed and the controller
    # chases it back out. Keys must match the training-time cond_steps (0, -1).
    cond = {
        0: np.asarray(observation, dtype=np.float32),
        cfg.horizon - 1: np.array([target[0], target[1], 0.0, 0.0], dtype=np.float32),
    }

    # The device comes from the MODEL, not from cfg: cfg came out of the checkpoint, so cfg.device
    # is where *training* ran, while eval resolves its own device and only moves the weights.
    device = str(next(diffusion.parameters()).device)
    _, observations = plan(diffusion, normalizer, cond, batch_size=cfg.plan_batch_size, device=device)
    waypoints = observations[0]

    total_reward, steps = 0.0, 0
    for t in range(env.max_episode_steps):
        state = env.state_vector().copy()
        if t + 1 < len(waypoints):
            next_waypoint = waypoints[t + 1]
        else:
            # Plan exhausted: hold the final position at zero velocity, which turns the PD
            # controller into a brake. Copy, because `waypoints` is also returned as 'plan'.
            next_waypoint = waypoints[-1].copy()
            next_waypoint[2:] = 0

        action = (next_waypoint[:2] - state[:2]) + (next_waypoint[2:] - state[2:])
        observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        steps = t + 1
        if terminal:
            break

    # d4rl's normalized score is a fraction; the paper's table is this x 100.
    return {'score': float(env.get_normalized_score(total_reward)), 'return': float(total_reward),
            'steps': int(steps), 'plan': waypoints}


def plot_plan(env, observations, path):
    """Save a top-down PNG of an xy path over the maze walls.

    maze2d debugging is visual: a plan that clips a wall or teleports across one is obvious here
    and invisible in the loss curve.
    """
    # Lazily imported and forced to Agg: compute nodes have no display, and importing this module
    # should not pick a GUI backend.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    maze = np.asarray(env.maze_arr)
    xy = np.asarray(observations)[:, :2]

    plt.figure(figsize=(5, 5))
    # maze_arr is indexed [row, col] and a maze2d position is (row, col) in that same grid, while
    # imshow puts index (r, c) at (x=c, y=r) -- hence x=obs[1], y=obs[0]. Swapping them gives a
    # transposed maze that still looks plausible, which is why it is worth a comment.
    plt.imshow(maze == WALL, cmap=plt.cm.binary, vmin=0, vmax=1)
    plt.plot(xy[:, 1], xy[:, 0], c='black', lw=1, zorder=10)
    plt.scatter(xy[:, 1], xy[:, 0], c=plt.cm.jet(np.linspace(0, 1, len(xy))), s=12, zorder=20)
    target = np.asarray(env._target)
    plt.scatter([target[1]], [target[0]], marker='*', s=250, c='red', zorder=30)
    plt.axis('off')

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path
