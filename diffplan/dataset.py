"""Normalization and the windowed trajectory dataset shared by both tasks."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

# Observations and actions are normalized independently, as upstream does.
KEYS = ('observations', 'actions')


class Normalizer:
    """Per-dimension affine map of [min, max] onto [-1, 1], then clip (the paper's LimitsNormalizer).

        z = (x - loc) / scale   with   loc = (min + max) / 2,  scale = (max - min) / 2
    """

    def __init__(self, loc, scale):
        self.loc = loc      # dict: key -> [dim] float32
        self.scale = scale  # dict: key -> [dim] float32, strictly positive

    @classmethod
    def fit(cls, episodes):
        assert len(episodes) > 0, 'cannot fit a normalizer on zero episodes'
        loc, scale = {}, {}
        for key in KEYS:
            # Streaming min/max rather than np.concatenate: a concatenated copy of the furniture
            # observations would double peak memory for no reason.
            mins = maxs = None
            for ep in episodes:
                x = np.asarray(ep[key], dtype=np.float64)
                if x.shape[0] == 0:
                    continue
                lo, hi = x.min(axis=0), x.max(axis=0)
                mins, maxs = (lo, hi) if mins is None else (np.minimum(mins, lo), np.maximum(maxs, hi))
            assert mins is not None, 'no transitions found while fitting the normalizer'
            # Clamp the SPAN, so a constant channel cannot divide by zero. Upstream's
            # SafeLimitsNormalizer writes `self.mins -= eps` where it means `self.mins[i] -= eps`,
            # and so widens every dimension once per constant dimension found.
            loc[key] = ((mins + maxs) / 2.0).astype(np.float32)
            scale[key] = (np.maximum(maxs - mins, 1e-6) / 2.0).astype(np.float32)
        return cls(loc, scale)

    def normalize(self, x, key):
        loc, scale = self._stats(key)
        z = (np.asarray(x, dtype=np.float32) - loc) / scale
        return np.clip(z, -1.0, 1.0).astype(np.float32)

    def unnormalize(self, x, key):
        loc, scale = self._stats(key)
        # Clip first: the model can emit slightly out-of-range values, and those must not be
        # amplified back into raw units.
        z = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
        return (z * scale + loc).astype(np.float32)

    def unnormalize_torch(self, x, key):
        """unnormalize() wherever the tensor already lives, bit-identical to the numpy path.

        clamp is exact and float32 multiply/add are correctly rounded, so CPU and GPU agree
        exactly. Exists because plan() unnormalizes a [plan_batch_size, H, dim] block ~90 times
        per episode, and single-threaded numpy dominated the rollout on a CPU-contended node.
        """
        loc, scale = self._stats(key)
        cache = self.__dict__.setdefault('_torch_cache', {})
        ck = (key, str(x.device), str(x.dtype))
        if ck not in cache:
            cache[ck] = (torch.as_tensor(loc, device=x.device, dtype=x.dtype),
                         torch.as_tensor(scale, device=x.device, dtype=x.dtype))
        loc_t, scale_t = cache[ck]
        return torch.clamp(x, -1.0, 1.0) * scale_t + loc_t

    def _stats(self, key):
        if key not in self.loc:
            raise KeyError('unknown normalizer key %r (have %s)' % (key, sorted(self.loc)))
        return self.loc[key], self.scale[key]

    def state_dict(self):
        # Plain numpy only, so this drops straight into the training checkpoint.
        return {'loc': {k: np.asarray(v, dtype=np.float32) for k, v in self.loc.items()},
                'scale': {k: np.asarray(v, dtype=np.float32) for k, v in self.scale.items()}}

    @classmethod
    def from_state_dict(cls, d):
        return cls({k: np.asarray(v, dtype=np.float32) for k, v in d['loc'].items()},
                   {k: np.asarray(v, dtype=np.float32) for k, v in d['scale'].items()})

    def __repr__(self):
        return '[ Normalizer ] dims=%s' % {k: int(v.shape[-1]) for k, v in self.loc.items()}


class TrajectoryDataset(torch.utils.data.Dataset):
    """Fixed-horizon windows over a list of episodes.

    episodes: list of {'observations': [T, obs_dim] f32,
                       'actions':      [T, act_dim] f32,
                       'rewards':      [T] f32}

    Each item is {'x': [horizon, act_dim + obs_dim], 'cond': {step: observation}} plus, when a
    discount is given, 'value': the normalized discounted return-to-go.
    """

    def __init__(self, episodes, horizon, cond_steps=(0,), discount=None, normalizer=None,
                 pad_last=False):
        assert horizon > 0
        self.horizon = int(horizon)
        self.discount = discount
        self.pad_last = bool(pad_last)

        # cond_steps is plain DATA -- a tuple of ints, not a subclass hook like upstream's
        # GoalDataset.get_conditions -- precisely so it can be written into the checkpoint. Eval
        # then cannot disagree with training about the conditioning scheme. Negatives count from
        # the end: maze2d passes (0, -1) for start/goal inpainting, furniture (0,).
        self.cond_steps = tuple(int(h) % self.horizon for h in cond_steps)

        self.episodes = list(episodes)
        assert len(self.episodes) > 0, 'empty episode list'
        self.observation_dim = int(self.episodes[0]['observations'].shape[-1])
        self.action_dim = int(self.episodes[0]['actions'].shape[-1])
        self.transition_dim = self.action_dim + self.observation_dim
        self.normalizer = normalizer if normalizer is not None else Normalizer.fit(self.episodes)

        # Note the "+ 1": upstream's `range(max_start)` (datasets/sequence.py make_indices)
        # silently drops the last valid window of every episode.
        self.indices = []  # type: List[Tuple[int, int]]
        for i, ep in enumerate(self.episodes):
            T = len(ep['observations'])
            assert len(ep['actions']) == T, 'episode %d: len(actions) != len(observations)' % i
            assert len(ep['rewards']) == T, 'episode %d: len(rewards) != len(observations)' % i
            n_starts = T if self.pad_last else T - self.horizon + 1
            self.indices.extend((i, s) for s in range(max(0, n_starts)))
        assert len(self.indices) > 0, (
            'no windows: horizon %d is longer than every episode; use pad_last=True or a shorter '
            'horizon' % self.horizon)

        self.values = None  # type: Optional[np.ndarray]
        if self.discount is not None:
            self.values = self._compute_values()

    def _compute_values(self):
        """Discounted return-to-go from each window start, rescaled to [-1, 1].

        Over the REST OF THE EPISODE, not just the window: the window rarely contains the
        assembly reward, and a target that ignores everything past the horizon is not a value.
        The rescaling matters because upstream's fixed guide_scale=0.1 was tuned for locomotion
        returns of magnitude 100-400 while ours are O(1).

        Rewards are NOT padded even when pad_last=True: the episode is over, so the correct
        continuation is "no further reward".
        """
        gamma = float(self.discount)
        per_episode = []
        for ep in self.episodes:
            r = np.asarray(ep['rewards'], dtype=np.float64).reshape(-1)
            g, acc = np.zeros(len(r)), 0.0
            for t in range(len(r) - 1, -1, -1):
                acc = r[t] + gamma * acc
                g[t] = acc
            per_episode.append(g)
        v = np.array([per_episode[i][s] for i, s in self.indices], dtype=np.float32)
        return 2.0 * (v - v.min()) / max(float(v.max() - v.min()), 1e-6) - 1.0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, start = self.indices[idx]
        ep = self.episodes[ep_idx]
        rows = start + np.arange(self.horizon)
        if self.pad_last:
            # Clamping the gather index IS "repeat the final transition", with no padded copy of
            # the observation array. Furniture needs it: the terminal frames carry all of the
            # reward, and upstream's use_padding=False would discard exactly them (while its
            # use_padding=True zero-pads, and a zero vector is not a valid observation).
            rows = np.minimum(rows, len(ep['observations']) - 1)

        obs = self.normalizer.normalize(ep['observations'][rows], 'observations')
        act = self.normalizer.normalize(ep['actions'][rows], 'actions')
        # Channel order is [actions | observations] everywhere.
        batch = {
            'x': np.ascontiguousarray(np.concatenate([act, obs], axis=-1), dtype=np.float32),
            'cond': {h: np.ascontiguousarray(obs[h], dtype=np.float32) for h in self.cond_steps},
        }
        if self.values is not None:
            batch['value'] = self.values[idx]
        return batch
