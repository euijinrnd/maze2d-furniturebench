"""Merge the per-shard eval_<i>of<n>.json files into one headline number.

scripts/eval.py writes one JSON per shard, each summarizing only that shard. Averaging the shard
means is wrong whenever they hold different episode counts, so this re-runs the same summarize()
over the concatenated per-episode rows -- exactly what an unsharded run would have computed.

    python scripts/aggregate_eval.py logs/one_leg/H32_T20_s0
    python scripts/aggregate_eval.py logs/one_leg/H32_T20_s0/eval_*of5.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# Repo root on the path, so this runs from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval import summarize  # noqa: E402


def collect_paths(args):
    paths = []
    for a in args:
        if os.path.isdir(a):
            paths.extend(sorted(glob.glob(os.path.join(a, 'eval_*of*.json'))))
        else:
            paths.extend(sorted(glob.glob(a)))
    return paths


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('paths', nargs='+', help='a run directory, or explicit eval_*of*.json files')
    p.add_argument('--output', default=None, help='where to write the merged json')
    args = p.parse_args(argv)

    paths = collect_paths(args.paths)
    if not paths:
        raise SystemExit('no eval_*of*.json found in %s' % ' '.join(args.paths))

    rows = []
    episodes = []
    checkpoints = set()
    steps = set()
    for path in paths:
        with open(path) as f:
            d = json.load(f)
        rows.extend(d['per_episode'])
        episodes.extend(d.get('episodes', []))
        checkpoints.add(d.get('checkpoint'))
        steps.add(d.get('step'))
        print('[agg] %-60s %3d episodes' % (os.path.basename(path), len(d['per_episode'])))

    # Silent mixing of two checkpoints (say a resumed run that rewrote state_latest.pt
    # mid-eval) would produce a number that belongs to neither. Refuse rather than average.
    if len(checkpoints) > 1:
        raise SystemExit('shards disagree on checkpoint: %s' % sorted(checkpoints))
    if len(steps) > 1:
        raise SystemExit('shards disagree on training step: %s' % sorted(steps))

    dupes = len(episodes) - len(set(episodes))
    if dupes:
        raise SystemExit(
            'episode indices overlap across shards (%d duplicates) -- are these shards of '
            'the same --episode-shard i/n run?' % dupes)

    stats = summarize(rows)
    out = {
        'checkpoint': checkpoints.pop(),
        'step': steps.pop(),
        'n_shards': len(paths),
        'episodes': sorted(episodes),
        'stats': stats,
    }
    path = args.output or os.path.join(os.path.dirname(os.path.abspath(paths[0])), 'eval_all.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print('[agg] %d shards, %d episodes' % (len(paths), len(rows)))
    print('[agg] %s' % json.dumps(stats, sort_keys=True))
    if 'success_mean' in stats:
        # The headline for one_leg, as a percentage to match the paper's 56%.
        print('[agg] SUCCESS RATE: %.1f%% over %d episodes'
              % (100.0 * stats['success_mean'], stats['n_episodes']))
    if 'normalized_score' in stats:
        print('[agg] NORMALIZED SCORE: %.1f' % stats['normalized_score'])
    print('[agg] wrote %s' % path)
    return out


if __name__ == '__main__':
    main()
