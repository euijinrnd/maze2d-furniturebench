"""FurnitureBench one_leg offline data: pkl demos -> a flat frame store -> trajectory windows.

No isaacgym here, so training runs on any GPU node. On disk (`scripts/build_furniture_dataset.py`):

    <data_dir>/images.npy    uint8 [N, 2, 224, 224, 3]   (wrist, front) -- memmapped, never loaded whole
    <data_dir>/frames.npz    robot_state [N, 14] f32, actions [N, 8] f32 (raw deltas), rewards [N] f32,
                             ep_start [E + 1] int64, success [E] bool

Trajectory channels are low-dimensional and rotation-continuous:
    observation (16) = [ee_pos 3 | ee_rot6d 6 | ee_pos_vel 3 | ee_ori_vel 3 | gripper_width 1]
    action      (10) = [target_pos 3 | target_rot6d 6 | gripper +-1]        (ABSOLUTE ee target)
Images enter the model only as conditioning at the window's first frame.
"""
from __future__ import annotations

import numpy as np
import torch

from diffplan.dataset import TrajectoryDataset

OBS_DIM, ACT_DIM = 16, 10


# --------------------------------------------------------------------------- rotations (xyzw)

def quat_mul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def quat_conj(q):
    return q * np.array([-1, -1, -1, 1], dtype=q.dtype)


def quat_to_mat(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def mat_to_quat(m):
    """Rotation matrix -> unit quaternion (xyzw), w >= 0. Shepperd's method, vectorized."""
    m = np.asarray(m, dtype=np.float64)
    t = np.trace(m, axis1=-2, axis2=-1)
    d = np.stack([m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], -1)
    # Four candidates keyed on which of (w, x, y, z) has the largest square.
    q = np.stack([
        np.stack([m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1], 1 + t], -1),
        np.stack([1 + 2 * d[..., 0] - t, m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0], m[..., 2, 1] - m[..., 1, 2]], -1),
        np.stack([m[..., 0, 1] + m[..., 1, 0], 1 + 2 * d[..., 1] - t, m[..., 1, 2] + m[..., 2, 1], m[..., 0, 2] - m[..., 2, 0]], -1),
        np.stack([m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1], 1 + 2 * d[..., 2] - t, m[..., 1, 0] - m[..., 0, 1]], -1),
    ], axis=-2)  # [..., 4 candidates, 4]
    best = np.argmax(np.stack([1 + t, 1 + 2 * d[..., 0] - t, 1 + 2 * d[..., 1] - t, 1 + 2 * d[..., 2] - t], -1), -1)
    q = np.take_along_axis(q, best[..., None, None], axis=-2)[..., 0, :]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return (q * np.where(q[..., 3:] < 0, -1.0, 1.0)).astype(np.float32)


def mat_to_rot6d(m):
    return m[..., :2, :].reshape(*m.shape[:-2], 6)


def rot6d_to_mat(r):
    a, b = r[..., :3], r[..., 3:]
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    b = b - (a * b).sum(-1, keepdims=True) * a
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    return np.stack([a, b, np.cross(a, b)], axis=-2)


# --------------------------------------------------------------------------- state / action layout

def encode_state(robot_state):
    """[T, 14] raw robot_state -> [T, 16] with the quaternion replaced by rot6d."""
    s = np.asarray(robot_state, dtype=np.float32)
    return np.concatenate([s[..., :3], mat_to_rot6d(quat_to_mat(s[..., 3:7])), s[..., 7:]], -1)


def encode_action(robot_state, delta):
    """Recorded delta action -> absolute ee target [T, 10].

    The env sets its OSC goal to (ee_pos + dpos, ee_quat x dquat) (furniture_sim_env.py step()),
    so the target is exact given the pose the delta was computed against.
    """
    s, d = np.asarray(robot_state, dtype=np.float32), np.asarray(delta, dtype=np.float32)
    quat = quat_mul(s[..., 3:7], d[..., 3:7])
    return np.concatenate([s[..., :3] + d[..., :3], mat_to_rot6d(quat_to_mat(quat)), d[..., 7:8]], -1)


def decode_action(target, robot_state):
    """Absolute target (10,) -> the env's delta action (8,) against the LIVE ee pose."""
    s = np.asarray(robot_state, dtype=np.float32)
    dquat = quat_mul(quat_conj(s[3:7]), mat_to_quat(rot6d_to_mat(target[3:9])))
    grip = 1.0 if target[9] > 0 else -1.0  # the env ignores |grip| < 0.019; L2 regresses toward 0
    return np.concatenate([target[:3] - s[:3], dquat, [grip]]).astype(np.float32)


# --------------------------------------------------------------------------- loading

def load_episodes(data_dir, n_success=100, n_failure=0):
    """-> list of {'observations' [T,16], 'actions' [T,10], 'rewards' [T], 'frames' [T] (row in images.npy)}.

    The failure demos stay on disk and remain loadable (`--n_failure 50` reproduces HDFlow's
    split), but they are excluded by default: every one of them is a 600-step collector timeout,
    and mixing them into the prior halved the success rate.
    """
    f = np.load('%s/frames.npz' % data_dir)
    starts, success = f['ep_start'], f['success']
    ids = np.flatnonzero(success)[:n_success].tolist() + np.flatnonzero(~success)[:n_failure].tolist()
    assert len(ids) == n_success + n_failure, 'only %d/%d success/failure episodes on disk' % (
        success.sum(), (~success).sum())
    episodes = []
    for i in ids:
        sl = slice(starts[i], starts[i + 1])
        episodes.append({
            'observations': encode_state(f['robot_state'][sl]),
            'actions': encode_action(f['robot_state'][sl], f['actions'][sl]),
            'rewards': f['rewards'][sl].astype(np.float32),
            'frames': np.arange(sl.start, sl.stop),
        })
    print('[furniture_data] %d episodes (%d success), %d transitions' % (
        len(ids), n_success, sum(len(e['rewards']) for e in episodes)))
    return episodes


class FurnitureDataset(TrajectoryDataset):
    """TrajectoryDataset plus the two uint8 camera images at each window's first frame."""

    def __init__(self, data_dir, episodes, **kw):
        super().__init__(episodes, **kw)
        # Pull the used frames into RAM once (~24 GB for 150 episodes): random reads from the
        # memmap on Lustre cost more per batch than the whole training step.
        frames = np.concatenate([ep['frames'] for ep in episodes])
        self.images = np.load('%s/images.npy' % data_dir, mmap_mode='r')[frames]
        self.offsets = np.concatenate([[0], np.cumsum([len(ep['frames']) for ep in episodes])])

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        ep_idx, start = self.indices[idx]
        batch['images'] = torch.from_numpy(self.images[self.offsets[ep_idx] + start])
        return batch
