"""Evaluate a Diffuser checkpoint: roll out episodes, print and JSON-dump the stats.

    python scripts/eval.py --checkpoint logs/maze2d-umaze-v1/H128_T64_s0/state_latest.pt
    python scripts/eval.py --checkpoint logs/one_leg/H32_T20_s0/state_latest.pt --episode-shard 3/10

The checkpoint is self-contained (EMA weights + normalizer stats + config), so nothing here
touches the dataset. Per-episode rows go to eval[<tag>][_iofn].json next to the checkpoint;
scripts/aggregate_eval.py merges shards.

Value guidance is OPT-IN (`--guide`): on one_leg it costs ~15 points, so the reported numbers are
unguided. It also wants a population to rank, hence `--override plan_batch_size=64` with it.
"""
from __future__ import annotations

try:
    import isaacgym  # noqa: F401  must precede torch
except ImportError:
    pass

import argparse
import dataclasses
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffplan.config import _parse_bool  # noqa: E402
from diffplan.dataset import Normalizer  # noqa: E402
from scripts.train import build_models, is_maze2d  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--checkpoint', required=True, help='the HIGH level, for a hierarchical run')
    p.add_argument('--low-checkpoint', default=None,
                   help='hierarchical runs; default: the sibling low/ of --checkpoint')
    p.add_argument('--n-episodes', type=int, default=None, help='default: cfg.n_eval_episodes')
    p.add_argument('--episode-shard', default=None, help='"i/n", 0 <= i < n')
    p.add_argument('--seed', type=int, default=0, help='episode e uses seed + e')
    p.add_argument('--guide', action='store_true', help='use the value model in the checkpoint')
    p.add_argument('--plot', action='store_true', help='maze2d only: save plan pngs')
    p.add_argument('--output-tag', default='', help='inserted into the json name: eval<tag>_<i>of<n>.json')
    p.add_argument('--override', nargs='*', default=[], metavar='KEY=VALUE',
                   help='eval-time config overrides, e.g. replan_every=8 plan_batch_size=64')
    return p.parse_args(argv)


def load_checkpoint(path, device, use_guide=False):
    """-> (step, cfg, diffusion, normalizer, guide or None), all from the checkpoint's own config."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    normalizer = Normalizer.from_state_dict(ckpt['normalizer'])
    obs_dim, act_dim = len(normalizer.loc['observations']), len(normalizer.loc['actions'])
    diffusion, guide = build_models(cfg, obs_dim, act_dim)
    diffusion.load_state_dict(ckpt['ema'])
    diffusion.to(device).eval()
    if guide is not None and use_guide and ckpt.get('value_ema') is not None:
        guide.load_state_dict(ckpt['value_ema'])
        guide.to(device).eval()
    else:
        guide = None
    return ckpt['step'], cfg, diffusion, normalizer, guide


def load_low_level(args, cfg, normalizer, device):
    """-> the low-level diffuser paired with a hierarchical --checkpoint, or None.

    A hierarchical run trains its two levels as two jobs, so the single-checkpoint pairing
    invariant has to be re-established here: the levels must agree on K, on the number of subgoals
    and on the normalizer, or the subgoals the high level emits mean something else to the low one.
    """
    if not cfg.level:
        return None
    assert cfg.level == 'high', '--checkpoint must be the high level of a hierarchical run'
    assert not args.guide, 'plan_hd does not use guidance; see diffplan/diffusion.py'
    run = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
    path = args.low_checkpoint or os.path.join(run, 'low', os.path.basename(args.checkpoint))
    _, low_cfg, low, low_norm, _ = load_checkpoint(path, device)
    assert (low_cfg.level, low_cfg.jump, low_cfg.n_subgoal) == ('low', cfg.jump, cfg.n_subgoal), (
        'level mismatch: high is HD%dx%d, %s is %r HD%dx%d'
        % (cfg.jump, cfg.n_subgoal, path, low_cfg.level, low_cfg.jump, low_cfg.n_subgoal))
    assert all(np.array_equal(getattr(low_norm, a)[k], getattr(normalizer, a)[k])
               for a in ('loc', 'scale') for k in normalizer.loc), (
        'the two levels were fitted on different data: %s' % path)
    print('[ eval ] low level: %s' % path, flush=True)
    return low


def summarize(rows):
    stats = {'n_episodes': len(rows)}
    for k in rows[0] if rows else ():
        v = np.array([r[k] for r in rows], dtype=np.float64)
        stats[k + '_mean'], stats[k + '_std'] = float(v.mean()), float(v.std())
    if 'score_mean' in stats:
        stats['normalized_score'] = 100.0 * stats['score_mean']
    return stats


def main(argv=None):
    args = parse_args(argv)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    step, cfg, diffusion, normalizer, guide = load_checkpoint(args.checkpoint, device, use_guide=args.guide)
    for kv in args.override:
        k, v = kv.split('=')
        cast = _parse_bool if isinstance(getattr(cfg, k), bool) else type(getattr(cfg, k))
        cfg = dataclasses.replace(cfg, **{k: cast(v)})
    low = load_low_level(args, cfg, normalizer, device)
    print('[ eval ] step %s | %s | guide %s' % (step, cfg, guide is not None), flush=True)

    if is_maze2d(cfg):
        from diffplan.tasks.maze2d import make_env, plot_plan, rollout
        env = make_env(cfg.env_id)
    else:
        from diffplan.tasks.furniture import OneLegEnv, rollout
        env = OneLegEnv(cfg)

    indices = list(range(args.n_episodes or cfg.n_eval_episodes))
    if args.episode_shard:
        i, n = map(int, args.episode_shard.split('/'))
        indices = indices[i::n]

    outdir = os.path.dirname(os.path.abspath(args.checkpoint))
    rows = []
    for e in indices:
        torch.manual_seed(args.seed + e)
        np.random.seed(args.seed + e)
        if is_maze2d(cfg):
            result = rollout(env, diffusion, normalizer, cfg, seed=args.seed + e, low=low)
            if args.plot:
                plot_plan(env, result['plan'], os.path.join(outdir, 'plan_%d.png' % e))
        else:
            result = rollout(env, diffusion, normalizer, cfg, guide=guide, low=low)
        rows.append({k: float(v) for k, v in result.items() if np.isscalar(v)})
        print('[ eval ] episode %d | %s' % (e, ' '.join('%s %.3f' % kv for kv in sorted(rows[-1].items()))), flush=True)

    stats = summarize(rows)
    suffix = '' if args.episode_shard is None else '_%s' % args.episode_shard.replace('/', 'of')
    path = os.path.join(outdir, 'eval%s%s.json' % (args.output_tag, suffix))
    with open(path, 'w') as f:
        json.dump({'checkpoint': os.path.abspath(args.checkpoint), 'step': step, 'episodes': indices,
                   'per_episode': rows, 'stats': stats, 'override': args.override, 'guide': guide is not None},
                  f, indent=2, sort_keys=True)
    print('[ eval ] %s\n[ eval ] wrote %s' % (json.dumps(stats, sort_keys=True), path))


if __name__ == '__main__':
    main()
