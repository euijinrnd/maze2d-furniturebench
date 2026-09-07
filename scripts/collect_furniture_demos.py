"""Collect scripted FurnitureSim demonstrations, KEEPING the failures (the value model's negatives).

    python scripts/collect_furniture_demos.py --out-dir data/sim --seed 0 --num-demos 25

--num-demos counts SUCCESSES; the scripted policy succeeds ~50% of the time on one_leg/low, so
failures accrue for free. Shards need distinct seeds (initial part placement comes from the
global numpy RNG) and get their own subdirectory (DataCollector names files by the second).
resize_sim_img is deliberately left False so the collector applies the same resize/resize_crop
the eval wrapper does (diffplan.tasks.furniture.OneLegEnv).
"""
import isaacgym  # noqa: F401  must precede torch

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from furniture_bench.data.data_collector import DataCollector

from diffplan.config import CONFIGS


def main():
    cfg = CONFIGS['one_leg']
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--furniture', default=cfg.furniture)
    p.add_argument('--randomness', default=cfg.randomness, choices=['low', 'med', 'high'])
    p.add_argument('--num-demos', type=int, default=25, help='target number of SUCCESSES')
    p.add_argument('--out-dir', default='data/sim')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_path = Path(args.out_dir) / args.furniture / ('seed%d' % args.seed)
    data_path.mkdir(parents=True, exist_ok=True)

    collector = DataCollector(
        is_sim=True, data_path=str(data_path), device_interface=None, furniture=args.furniture, headless=True,
        draw_marker=False, manual_label=False, scripted=True, randomness=args.randomness, compute_device_id=0,
        graphics_device_id=0, pkl_only=True, save_failure=True, num_demos=args.num_demos)
    collector.collect()
    print('[collect] %s: successes=%d failures=%d' % (data_path, collector.num_success, collector.num_fail))


if __name__ == '__main__':
    main()
