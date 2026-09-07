"""Flat dataclass configs plus a two-pass argparse loader.

Pure stdlib on purpose: `scripts/eval.py` reads a config to decide whether the run needs a
simulator at all, BEFORE importing isaacgym/torch. A torch import here would defeat that order.

`Config` holds what both tasks share; each task subclass restates only what differs.
"""

from __future__ import annotations

import argparse
import dataclasses
import typing
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class Config:
    name: str = ''
    env_id: str = ''

    # model / diffusion
    horizon: int = 128
    n_diffusion_steps: int = 64
    dim: int = 32
    dim_mults: Tuple[int, ...] = (1, 4, 8)
    predict_epsilon: bool = False  # the model outputs x0 directly, as in the paper's maze2d branch
    clip_denoised: bool = True
    loss_type: str = 'l2'
    action_weight: float = 1.0

    # dataset. The normalizer maps [min, max] -> [-1, 1] per dimension, so `clip` is both its
    # bound and the x0 clamp: clamping x0 to anything other than the range the data occupies
    # would be meaningless.
    clip: float = 1.0
    cond_steps: Tuple[int, ...] = (0,)  # timesteps to inpaint; negatives count from the end
    pad_last: bool = False

    # A value model is trained only when discount is not None.
    discount: Optional[float] = None

    # training
    lr: float = 2e-4
    cosine_lr: bool = False  # upstream trains at a constant lr
    batch_size: int = 32
    grad_accum: int = 2
    ema_decay: float = 0.995
    ema_start: int = 2000
    ema_every: int = 10
    n_train_steps: int = 2_000_000
    log_every: int = 100
    save_every: int = 10_000       # rewrites state_latest.pt in place; no disk growth
    snapshot_every: int = 100_000  # archival state_<step>.pt, so kept rare
    num_workers: int = 1

    # eval
    n_eval_episodes: int = 100
    plan_batch_size: int = 1

    # logging. A "probe" is a mid-training rollout of a few fixed-seed episodes with the EMA
    # weights (build_probe in scripts/train.py). Its cadence is wall clock, not steps, because
    # steps/second varies ~3x across the mazes and again across GPU models.
    use_wandb: bool = True
    wandb_project: str = 'diffplan'
    wandb_entity: Optional[str] = None  # None -> the account default entity
    probe_every_seconds: float = 3600.0
    probe_episodes: int = 4

    # misc
    seed: int = 0
    device: str = 'cuda'
    logbase: str = 'logs'
    smoke: bool = False


@dataclass
class Maze2dConfig(Config):
    """D4RL Maze2D: goal-conditioned inpainting, plan once and follow open loop.

    Values verified against `git show origin/maze2d:config/maze2d.py`: the 'base' dict there is
    horizon 256 / T 256 (what *medium* uses, having no override block), umaze 128/64, large 384/256.
    """

    cond_steps: Tuple[int, ...] = (0, -1)  # inpaint start AND goal
    pad_last: bool = False  # episodes are long; the tail windows carry nothing special


@dataclass
class OneLegConfig(Config):
    """FurnitureBench one_leg @ randomness=low: image-conditioned, closed-loop, optionally guided.

    Trajectory channels are [absolute ee target (10) | proprio (16)]; the two 224x224 camera frames
    at the window start condition the U-Net through an end-to-end ResNet18. Diffusion
    hyperparameters follow the official locomotion config (horizon 32 / T 20 / a_0 x10 / x0
    prediction; guidance n_guide_steps 2 / scale 0.1 / t_stopgrad 2 / discount 0.997).
    """

    name: str = 'one_leg'
    env_id: str = 'one_leg'

    horizon: int = 32
    n_diffusion_steps: int = 20
    dim: int = 128
    action_weight: float = 10.0
    encoder: str = 'resnet18'  # or 'dinov2' (frozen ViT-B/14, HDFlow's encoder; use --crop 196)
    crop: int = 200  # random (train) / center (eval) crop of the 224x224 frames

    # dataset
    data_dir: str = 'data/one_leg_low'
    n_success: int = 100  # HDFlow B.1: 100 successful + 50 failed demos per task and randomness
    n_failure: int = 0  # the failures are collector timeouts and hurt the prior (docs/one_leg.md)
    cond_steps: Tuple[int, ...] = (0,)  # only the current state is known at plan time
    pad_last: bool = True  # the terminal frames are exactly where the single reward lives

    # value model / guidance. discount 0.997 spans [0.17, 1] over a 700-step success; failures are 0.
    discount: Optional[float] = 0.997
    guide_scale: float = 0.1
    t_stopgrad: int = 2
    n_guide_steps: int = 2

    # training
    lr: float = 1e-4
    batch_size: int = 64
    grad_accum: int = 1
    n_train_steps: int = 200_000
    num_workers: int = 4

    # eval
    max_steps: int = 700
    plan_batch_size: int = 1  # guidance is opt-in (--guide) and wants 64 samples to rank
    replan_every: int = 2  # execute this many planned targets open loop, then replan

    # simulator
    furniture: str = 'one_leg'
    randomness: str = 'low'


CONFIGS = {
    'maze2d-umaze': Maze2dConfig(
        name='maze2d-umaze', env_id='maze2d-umaze-v1', horizon=128, n_diffusion_steps=64,
    ),
    'maze2d-medium': Maze2dConfig(
        name='maze2d-medium', env_id='maze2d-medium-v1', horizon=256, n_diffusion_steps=256,
    ),
    'maze2d-large': Maze2dConfig(
        name='maze2d-large', env_id='maze2d-large-v1', horizon=384, n_diffusion_steps=256,
    ),
    'one_leg': OneLegConfig(),
}


def run_name(cfg) -> str:
    """Directory name for a run: '<env_id>/H<horizon>_T<T>_s<seed>'."""
    return '{}/H{}_T{}_s{}'.format(cfg.env_id, cfg.horizon, cfg.n_diffusion_steps, cfg.seed)


# --------------------------------------------------------------------------------- CLI

# --smoke overrides, applied BEFORE the user's explicit flags so that `--smoke --horizon 16` does
# what it looks like. Sized to train and eval end-to-end on CPU in a few seconds; horizon must
# stay divisible by 2 ** (len(dim_mults) - 1) == 4. Keys absent from the dataclass are skipped.
_SMOKE = {
    'horizon': 8,
    'n_diffusion_steps': 4,
    'n_train_steps': 20,
    'batch_size': 2,
    'grad_accum': 1,
    'ema_start': 2,
    'log_every': 5,
    'save_every': 20,
    'snapshot_every': 20,
    'num_workers': 0,
    'n_eval_episodes': 1,
    'plan_batch_size': 2,
    'device': 'cpu',
    # No wandb run, but the probe SHOULD still fire (probe_every_seconds triggers one on the
    # first step), so --smoke covers the rollout-and-plot path end to end.
    'use_wandb': False,
    'probe_episodes': 1,
    'n_success': 2,
    'n_failure': 1,
    'max_steps': 10,
    'dim': 16,
}

_TRUE = ('1', 'true', 't', 'yes', 'y')
_FALSE = ('0', 'false', 'f', 'no', 'n')


def _parse_bool(s: str) -> bool:
    """argparse type for bools: 'store_true' cannot express --flag false."""
    low = s.lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise argparse.ArgumentTypeError("expected a boolean, got '{}'".format(s))


def _base_type(hint) -> Any:
    """Strip Optional[...] down to the payload type. Tuples are reported as tuple."""
    origin = typing.get_origin(hint)
    if origin is typing.Union:
        args = [a for a in typing.get_args(hint) if a is not type(None)]  # noqa: E721
        if len(args) != 1:
            raise TypeError('unsupported union in config annotation: {}'.format(hint))
        return _base_type(args[0])
    if origin is tuple:
        return tuple
    return hint


def load_config(argv=None):
    """Two-pass argparse: --config picks the dataclass, then its fields become flags.

    Pass 1 only looks for --config, because the set of valid flags depends on which dataclass we
    end up with. Pass 2 declares one flag per field with default=None as a "not given" sentinel,
    so an override is distinguishable from a value equal to the default.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        '--config', default='maze2d-umaze', choices=sorted(CONFIGS),
        help='which task config to start from',
    )
    known, _ = pre.parse_known_args(argv)
    cfg = CONFIGS[known.config]

    hints = typing.get_type_hints(type(cfg))
    parser = argparse.ArgumentParser(parents=[pre], description='diffplan training / eval config')
    tuple_item_type = {}
    for f in dataclasses.fields(cfg):
        hint = hints[f.name]
        base = _base_type(hint)
        flag = '--' + f.name
        current = getattr(cfg, f.name)
        if base is bool:
            # nargs='?' lets both `--smoke` and `--smoke false` work.
            parser.add_argument(flag, type=_parse_bool, nargs='?', const=True, default=None,
                                help='(bool, default {})'.format(current))
        elif base is tuple:
            args = [a for a in typing.get_args(hint) if a is not Ellipsis]
            item = args[0] if args else int
            tuple_item_type[f.name] = item
            parser.add_argument(flag, type=item, nargs='+', default=None,
                                help='(tuple, default {})'.format(current))
        else:
            parser.add_argument(flag, type=base, default=None,
                                help='({}, default {})'.format(base.__name__, current))

    args = parser.parse_args(argv)
    overrides = {}
    for f in dataclasses.fields(cfg):
        val = getattr(args, f.name)
        if val is None:
            continue
        if f.name in tuple_item_type:
            val = tuple(val)
        overrides[f.name] = val

    if overrides.get('smoke', cfg.smoke):
        smoke = {k: v for k, v in _SMOKE.items() if hasattr(cfg, k)}
        cfg = dataclasses.replace(cfg, smoke=True, **smoke)
    return dataclasses.replace(cfg, **overrides)


def config_to_dict(cfg) -> Dict[str, Any]:
    """Plain-dict view of a config, for logging / JSON.

    Trainer writes the dataclass INSTANCE into the .pt (diffplan.config imports no torch, so
    unpickling it never drags in a simulator, and eval just reads attributes). These two helpers
    are the escape hatch for the dict form; config_from_dict drops unknown keys, so a checkpoint
    written before a field existed -- or after one was removed -- still loads.
    """
    d = dataclasses.asdict(cfg)
    d['_class'] = type(cfg).__name__
    return d


def config_from_dict(d: Dict[str, Any]):
    """Inverse of config_to_dict, tolerant of fields added or removed since it was written."""
    cls = {'Maze2dConfig': Maze2dConfig, 'OneLegConfig': OneLegConfig}[d['_class']]
    names = set(f.name for f in dataclasses.fields(cls))
    return cls(**{k: (tuple(v) if isinstance(v, list) else v)
                  for k, v in d.items() if k in names})
