"""Flatten collected demo pkls into the frame store diffplan.tasks.furniture_data reads.

    python scripts/build_furniture_dataset.py --demo-dir data/sim/one_leg --out-dir data/one_leg_low

Every pkl holds T+1 observations and T actions; frame t is (obs[t], action[t], reward[t]).
Images are written straight into a memmapped uint8 array so the 20+ GB never sits in RAM.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

from furniture_bench.robot.robot_state import filter_and_concat_robot_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--demo-dir', required=True)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()

    pkls = sorted(Path(args.demo_dir).rglob('*.pkl'))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lengths = [len(pickle.load(open(p, 'rb'))['actions']) for p in pkls]
    ep_start = np.concatenate([[0], np.cumsum(lengths)])
    n = int(ep_start[-1])
    images = np.lib.format.open_memmap(out / 'images.npy', mode='w+', dtype=np.uint8, shape=(n, 2, 224, 224, 3))
    robot_state, actions, rewards, success = np.zeros((n, 14), np.float32), np.zeros((n, 8), np.float32), np.zeros(n, np.float32), []

    for i, path in enumerate(pkls):
        d = pickle.load(open(path, 'rb'))
        sl = slice(ep_start[i], ep_start[i + 1])
        obs = d['observations'][:lengths[i]]
        images[sl, 0] = np.stack([o['color_image1'] for o in obs])
        images[sl, 1] = np.stack([o['color_image2'] for o in obs])
        robot_state[sl] = np.stack([filter_and_concat_robot_state(o['robot_state']) for o in obs])
        actions[sl] = np.asarray(d['actions'])
        rewards[sl] = np.asarray(d['rewards'])
        success.append(bool(d['success']))
        print('[%d/%d] %s T=%d success=%s' % (i + 1, len(pkls), path.parent.name, lengths[i], success[-1]), flush=True)

    images.flush()
    np.savez(out / 'frames.npz', robot_state=robot_state, actions=actions, rewards=rewards,
             ep_start=ep_start, success=np.array(success))
    print('wrote %d frames, %d episodes (%d success) to %s' % (n, len(pkls), sum(success), out))


if __name__ == '__main__':
    main()
