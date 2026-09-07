# maze2d-furniturebench

Diffuser (Janner et al., 2022) reimplemented for D4RL Maze2D and FurnitureBench one_leg assembly.

## Overview

`diffplan` is a from-scratch, minimal reimplementation of **Diffuser** (Janner et al., 2022,
[arXiv:2205.09991](https://arxiv.org/abs/2205.09991)): a DDPM over whole trajectories
`x = [actions | observations]` of fixed horizon, denoised by a 1D temporal U-Net, where
planning *is* inpainting -- the known states are pinned at every reverse step. Two tasks share
one core:

* **D4RL Maze2D** conditions on `{0: start, H-1: goal}`, plans once, and tracks the predicted
  *states* with a hand-written PD controller (the diffused actions are discarded, as upstream does).
* **FurnitureBench one_leg @ randomness=low** conditions on `{0: current proprio}` plus the two
  camera frames (fed to the U-Net as a global context vector through an end-to-end ResNet18),
  executes the first `replan_every` planned absolute end-effector targets, replans, and can
  optionally rank / guide a batch of samples with a noise-aware return predictor.

Everything is flat modules and plain dataclasses, and every checkpoint is self-contained
(weights, EMA weights, normalizer statistics and config in one `.pt`).

## Module map

| Path | What it is |
| --- | --- |
| `diffplan/nn.py` | `TemporalUnet` (the x0 model), `TemporalValue` (the guide), both taking an optional context vector; `ImageEncoder` (ResNet18-GN + spatial softmax per camera). |
| `diffplan/diffusion.py` | `cosine_beta_schedule`, `apply_conditioning`, `GaussianDiffusion` (train / sample / guided sample, owns the encoder), `ReturnPredictor`, `plan()`. |
| `diffplan/dataset.py` | `Normalizer` (`[min, max] -> [-1, 1]`) and `TrajectoryDataset` (windowed episodes, the `cond` dict, discounted return-to-go). |
| `diffplan/trainer.py` | One loop that steps both models with EMA, one self-contained checkpoint. |
| `diffplan/config.py` | A shared `Config` dataclass with `Maze2dConfig` / `OneLegConfig` on top, the `CONFIGS` registry, `run_name`, two-pass argparse `load_config` (`--smoke` shrinks everything to CPU size). |
| `diffplan/tasks/maze2d.py` | d4rl re-segmentation, `make_env`, goal-conditioned open-loop `rollout`, `plot_plan`. |
| `diffplan/tasks/furniture_data.py` | Rotation utilities (quat / rot6d), the 16-d state and 10-d absolute-action layout, `load_episodes`, `FurnitureDataset` (adds the frame images). No simulator import. |
| `diffplan/tasks/furniture.py` | `OneLegEnv` (FurnitureSim with the collector's image pipeline) and the chunked closed-loop `rollout`. Imports isaacgym first. |
| `scripts/train.py`, `scripts/eval.py`, `scripts/aggregate_eval.py` | train; evaluate a checkpoint (`--episode-shard i/n`, `--override k=v`, `--guide`); merge the shards into one number. |
| `scripts/collect_furniture_demos.py`, `scripts/build_furniture_dataset.py` | scripted sim collection (failures kept), then pkls -> one memmapped frame store. |
| `tests/` | CPU-only invariants: layout, windows, normalizer, rotations, conditioning, guidance, a 400-step overfit. `pytest -q` runs in ~30 s. |
| `slurm/` | `train` / `eval` / `collect` / `cpu` job wrappers. |

## Install

```bash
bash env/create_env.sh                        # builds the `diffuser` conda env (python 3.8)
conda activate diffuser && source env/activate.sh
```

Python 3.8 is both floor and ceiling (IsaacGym Preview 4 ships no py>=3.9 binary;
furniture_bench pins `>3.7,<3.9`), and the install order inside `create_env.sh` is
load-bearing.

### Environment gotchas

Each of these cost a debugging cycle on Rocky Linux 9.4 / glibc 2.34 / driver 580.126.09:

* **`OMP_NUM_THREADS=4` is a ~430x speedup on CPU, not a micro-optimisation** (set by
  `env/activate.sh`). torch defaults its intra-op pool to the core count (52 here) and then
  spends essentially all of its time synchronizing threads, because a `TemporalUnet` activation
  is tiny. Measured on one training step, batch 32, horizon 8: 52 threads 30415 ms, 8 threads
  235 ms, **4 threads 71 ms**, 1 thread 125 ms. It also keeps us from taking over every core of
  a shared login node.
* **Build `gymtorch` on a GPU node, not the login node.** `import isaacgym.gymtorch` lazily
  compiles a torch C++ extension into `~/.cache/torch_extensions/py38_cu121/gymtorch/`. On a
  machine with no visible GPU it links CPU-only (`-ltorch_cpu`, no nvcc step), and torch's
  extension loader keys on the *source* hash, not the device config -- so a GPU node will
  silently reuse the CUDA-less build. If IsaacGym ever behaves as though there is no device,
  `rm -rf ~/.cache/torch_extensions/py38_cu121/gymtorch` and re-import on the GPU node.
* **`MUJOCO_PY_FORCE_CPU=1` is required** (set by `env/activate.sh`). `mujoco_py/builder.py:75`
  otherwise selects `LinuxGPUExtensionBuilder` because `/usr/lib/nvidia` exists, and that
  compiles `gl/eglshim.c` against `GL/glew.h`, a header that needs root to install. We never
  render MuJoCo -- maze2d eval only steps physics and `plot_plan` uses matplotlib.
* **`mujoco-py` must match the MuJoCo binary on disk.** This box has `~/.mujoco/mujoco210`, so
  the pin is `2.1.2.14`; `2.0.2.13` aborts its own `setup.py` looking for `mujoco200`. The
  version is also resolved during the build, so `MUJOCO_PY_MUJOCO_PATH` has to be exported
  *before* pip runs.
* `numpy` is pinned `<1.24` because `isaacgym/torch_utils.py:135` uses `np.float` as a *default
  argument*, so it is evaluated at import time.

Verified on the login node: python 3.8.20, numpy 1.23.5, gym 0.21.0, torch 2.4.1+cu121,
mujoco-py 2.1.2.14, d4rl 1.1, furniture_bench, isaacgym (`gym_38.so` + `gymapi` +
`gymtorch`), `maze2d-umaze-v1` dataset. Not yet verified: anything that needs a GPU --
constructing a `FurnitureSim` env, and every training or eval run.

## Maze2D

```bash
# train
python scripts/train.py --config maze2d-umaze
python scripts/train.py --config maze2d-medium
python scripts/train.py --config maze2d-large

# evaluate 100 episodes (--plot also writes plan_<i>.png next to the checkpoint)
python scripts/eval.py --checkpoint logs/maze2d-umaze-v1/H128_T64_s0/state_latest.pt

# seconds on CPU, no GPU: shrinks horizon / T / train steps / batch
python scripts/train.py --config maze2d-umaze --smoke
pytest -q tests/test_smoke.py

# on the cluster (suma_rtx4090 + base_qos; see the header of slurm/train.sbatch for the
# fallback order and for which partitions are unusable)
sbatch slurm/train.sbatch --config maze2d-umaze
```

### Watching a run

Training logs to Weights & Biases by default (`--use_wandb false` to turn it off; nothing in
`diffplan/` imports wandb, the trainer just takes any object with a `.log()` method).

Alongside the loss, every `probe_every_seconds` (default 3600) the trainer runs a **probe**: it
rolls out `probe_episodes` fixed-seed episodes with the EMA weights and logs
`probe/normalized_score` plus a top-down PNG of each plan, also written to
`logs/<run>/plans/step<N>_ep<i>.png`. Two reasons it is worth the wall clock:

* The loss curve cannot show you the two things most likely to be wrong — a plan that clips a
  wall, or an inverted normalizer. Both are obvious at a glance in the picture.
* `probe/normalized_score` is produced by the same rollout code as `scripts/eval.py` and scaled
  ×100 the same way, so it is in the paper's units. It is an estimate of the level, not a
  substitute for the headline number: it averages `probe_episodes` (4) fixed seeds against
  eval's 100, so early in training its standard error is tens of points, and because the seeds
  never change its plateau carries a small fixed bias. Quote `scripts/eval.py`, not the probe.

The cadence is wall-clock rather than step-based because steps/second varies ~3× across the
three maze sizes and again across GPU models, so a step count that probes hourly on `large`
would probe every few minutes on `umaze`.

A probe runs immediately at step 1, deliberately: it proves the rollout → plot → upload path
works in the first minute rather than failing an hour in. That first image comes from an
untrained model and is *supposed* to look like scribble.

## FurnitureBench one_leg @ low

```bash
# 1. collect scripted demos, keeping failures (4 shards x 25 successes ~ 100 success + ~100 failure)
sbatch --array=1-4 slurm/collect.sbatch --num-demos 25 --out-dir data/sim

# 2. flatten the pkls into data/one_leg_low/{images.npy, frames.npz}
sbatch slurm/cpu.sbatch scripts/build_furniture_dataset.py --demo-dir data/sim/one_leg --out-dir data/one_leg_low

# 3. train (checkpoints belong on /scratch2 -- /home has a real default quota)
sbatch slurm/train.sbatch --config one_leg --logbase /scratch2/$USER/maze2d-furniturebench/artifacts/logs/img128

# 4. evaluate 100 closed-loop episodes in 5 shards, then merge
sbatch --array=0-4 slurm/eval.sbatch --checkpoint logs/img128/one_leg/H32_T20_s0/state_latest.pt
python scripts/aggregate_eval.py logs/img128/one_leg/H32_T20_s0
```

Step 1 keeps the failures, and `--n_failure 50` in step 3 restores HDFlow's 100 + 50 split, but
the default trains on the successes alone. Value guidance is opt-in (`--guide`, with
`--override plan_batch_size=64` to give it a population to rank): both cost success rate here,
see `docs/one_leg.md`.

### What the model sees and does

* **Observation** (HDFlow B.1: RGB front + wrist, 14-d proprio, no part poses). The trajectory's
  observation channels are the 16-d proprio `[ee_pos | ee_rot6d | ee_pos_vel | ee_ori_vel |
  gripper_width]`; the two 224x224 frames at the window's first step go through a per-camera
  ResNet18 (GroupNorm, trained from scratch, 64-keypoint spatial softmax, random 200-px crop at
  train / center crop at eval) and enter every U-Net block alongside the time embedding.
* **Action**: the recorded 8-d deltas are converted to **absolute** end-effector targets
  `[pos | rot6d | grip]`. This is exact -- the simulator sets its OSC goal to
  `(ee_pos + dpos, ee_quat x dquat)` -- and the rollout re-derives a delta against the live pose
  at every step, so an open-loop chunk does not accumulate the drift that replaying deltas does. The gripper channel is binarized (the env ignores `|grip| < 0.019`).
* **Execution**: plan `horizon=32`, execute `replan_every=2` targets against the live pose, replan.
* **Data**: the 100 successful scripted demos of HDFlow's 100 + 50 split (`n_failure=0`). The
  failures are all 600-step collector timeouts and drag the prior down when mixed in.

`data/` is a symlink to `/scratch2/$USER/maze2d-furniturebench/data`. Demo pkls are ~150 MB
each (uncompressed image pairs); the frame store is a 32 GB uint8 memmap that `FurnitureDataset`
pulls into RAM for the episodes it uses (~24 GB), because random reads from Lustre cost more
per batch (280 ms) than the whole training step (115 ms on a 3090, ~8 it/s).

Evaluate one_leg on `suma_rtx4090` when possible: contact physics differs across GPU
generations enough that the scripted expert scores 5/10 on a 4090 and 2/10 on a 3090, and
`n >= 100` episodes -- per-episode outcomes are chaotic and a 20-episode number is noise.

## Results

| Benchmark | Metric | Diffuser target | Ours |
| --- | --- | --- | --- |
| maze2d-umaze-v1 | D4RL normalized score | 113.9 | 109.7 (95% CI contains the target) |
| maze2d-medium-v1 | D4RL normalized score | 121.5 | **128.7** |
| maze2d-large-v1 | D4RL normalized score | 123.0 | **136.0** |
| one_leg @ randomness=low | success rate over 100 episodes | 56% (HDFlow Table 1) | **38%** (2026-09-07) |

100 episodes each (`scripts/eval.py`, seeds 0-99), score x100 as the paper reports it.
The one_leg row is the image-conditioned pipeline above; `docs/one_leg.md` has the ablations,
including the DINOv2 encoder and the Diffusion-Policy training schedule, both worse. The earlier
R3M-feature pipeline peaked at 13% and was retired to
`/scratch2/$USER/maze2d-furniturebench/attic_r3m_pipeline_2026-09-06/`.

## Deviations from the official implementation

* **The asymmetric U-Net is kept on purpose.** The down path iterates `in_out` (3 stages for
  `dim_mults=(1,4,8)`) while the up path iterates `reversed(in_out[1:])` (2 stages), so the
  highest-resolution skip is pushed and never popped
  (`diffuser/models/temporal.py:89`, branch `origin/maze2d`). It looks like a bug and probably
  is one, but the published numbers came from exactly this network; symmetrizing it would
  change the parameter count and invalidate the comparison.
* **Upstream bugs we fixed.** `ValueFunction`'s hardcoded `fc_dim` is wrong for
  `dim_mults=(1,4,8)` (a hard `RuntimeError`), so `TemporalValue` measures the flattened size
  with a dry-run forward instead. `SafeLimitsNormalizer` widens *every* dimension once per
  constant dimension found (`self.mins -= eps` where `self.mins[i] -= eps` was meant); we
  clamp the span instead. The window indexer's `range(max_start)` drops the last valid window
  of every episode; we add the `+ 1` (and pad by repeating the final transition for furniture,
  never with zeros). `apply_conditioning`'s open-ended slice is bounded, so a dim mismatch is
  a loud error rather than a silent broadcast. Conditioning is re-applied to the model output
  only when `predict_epsilon=False`; upstream does it unconditionally, which inflates the
  reported epsilon-space loss. `p_mean_variance`'s `else: assert RuntimeError()` (a truthy
  object, so it never fires) is simply omitted. The value net masks the *observation* channels
  at t=0 -- pinned by inpainting, hence gradient-free -- which upstream left commented out at
  `temporal.py:214`, and masks only those, because `a_0` is what we actually execute.
* **T=20 for one_leg, taken from the locomotion config rather than maze2d's 256.** Closed-loop
  eval costs `T x 700 steps x 100 episodes` U-Net forwards: 1.4M (~2 h) at T=20 versus 18M
  (~26 h) at T=256.
* **Image conditioning and absolute actions for one_leg.** Diffuser is state-based; HDFlow's
  Diffuser baseline is image-based but unspecified. The context vector rides with the time
  embedding into every block (the same additive path Diffuser already uses for `t`), which is
  the smallest change that gives the planner eyes. Absolute rot6d targets replace the recorded
  deltas, which are per-step feedback terms that fail even when replayed from the exact state.
* Smaller, deliberate simplifications: no `Rearrange`-to-4D around `GroupNorm`
  (bit-identical); guidance is an `if` branch instead of a pluggable `sample_fn` package;
  `ReturnPredictor` does not subclass `GaussianDiffusion`; the value head is LayerNormed (an un-normalized trunk drove Mish to a hard zero at dim >= 128 and froze the guide); `batch_size` is an explicit
  argument to `sample()`; value targets are rescaled into [-1, 1] (upstream's fixed
  `scale=0.1` assumes returns of magnitude 100-400, ours are O(1)); and one checkpoint holds
  both models plus the normalizer, so a value model can never be paired with a mismatched
  diffusion model or normalizer.
