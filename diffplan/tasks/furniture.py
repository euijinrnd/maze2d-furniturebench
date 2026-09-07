"""FurnitureBench one_leg task: simulator wrapper and closed-loop rollout.

isaacgym MUST be imported before torch (it loads its own CUDA/torch shims and aborts otherwise),
which is why this module is the only one that touches the simulator and imports it first.
"""
from __future__ import annotations

import isaacgym  # noqa: F401  # isort:skip

import furniture_bench  # noqa: F401  # isort:skip -- registers the FurnitureSim* gym ids
import gym
import numpy as np
from furniture_bench.perception.image_utils import resize, resize_crop

from diffplan.diffusion import plan
from diffplan.tasks.furniture_data import decode_action, encode_state


class OneLegEnv:
    """FurnitureSim with the collector's exact observation pipeline: 14-d robot_state plus the
    wrist / front cameras resized to 224x224 (`resize` / `resize_crop`, as data_collector.py does).
    """

    def __init__(self, cfg):
        self.env = gym.make('FurnitureSim-v0', furniture=cfg.furniture, randomness=cfg.randomness, num_envs=1,
                            headless=True, max_env_steps=cfg.max_steps, act_rot_repr='quat', resize_img=False,
                            concat_robot_state=True, np_step_out=True)

    def _obs(self, obs):
        images = np.stack([resize(obs['color_image1'][0]), resize_crop(obs['color_image2'][0])])
        return obs['robot_state'][0].astype(np.float32), images

    def reset(self):
        return self._obs(self.env.reset())

    def step(self, action):
        obs, reward, done, _ = self.env.step(np.asarray(action, dtype=np.float32)[None])
        return self._obs(obs), float(np.reshape(reward, -1)[0]), bool(np.reshape(done, -1)[0])


def rollout(env, diffusion, normalizer, cfg, guide=None):
    """One closed-loop episode: plan H steps, execute `replan_every` absolute targets, repeat."""
    device = str(next(diffusion.parameters()).device)
    guide_kwargs = dict(guide_scale=cfg.guide_scale, t_stopgrad=cfg.t_stopgrad, n_guide_steps=cfg.n_guide_steps) if guide else {}
    (robot_state, images), total_reward, steps, done = env.reset(), 0.0, 0, False
    while steps < cfg.max_steps and not done:
        actions, _ = plan(diffusion, normalizer, {0: encode_state(robot_state)}, batch_size=cfg.plan_batch_size,
                          images=images, guide=guide, device=device, **guide_kwargs)
        for target in actions[0][:cfg.replan_every]:
            (robot_state, images), reward, done = env.step(decode_action(target, robot_state))
            total_reward += reward
            steps += 1
            if done or steps >= cfg.max_steps:
                break
    success = total_reward >= 1.0 - 1e-6  # one_leg pays 1.0 exactly once, at assembly
    print('[rollout] success=%s steps=%d' % (success, steps), flush=True)
    return {'success': bool(success), 'score': float(success), 'steps': steps}
