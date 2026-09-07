"""Train Diffuser on D4RL Maze2D or FurnitureBench one_leg.

    python scripts/train.py --config maze2d-umaze
    python scripts/train.py --config one_leg --data_dir data/one_leg_low
    python scripts/train.py --config maze2d-umaze --smoke        # seconds, on CPU

Everything the evaluator needs (EMA weights, normalizer stats, config) goes into a single
self-contained checkpoint, so eval never rebuilds the dataset.

This module deliberately never imports diffplan.tasks.furniture (isaacgym) or
diffplan.tasks.maze2d (d4rl/mujoco_py) at module scope: training on furniture needs no simulator
at all, so a training node needs neither installed. The task import happens inside build_*().
"""
from __future__ import annotations

import os

import numpy as np
import torch

from diffplan.config import config_to_dict, load_config, run_name
from diffplan.dataset import Normalizer, TrajectoryDataset
from diffplan.diffusion import GaussianDiffusion, ReturnPredictor
from diffplan.nn import ImageEncoder, TemporalUnet, TemporalValue
from diffplan.trainer import Trainer


def is_maze2d(cfg):
    """Configs are flat dataclasses; only OneLegConfig carries a `furniture` field."""
    return not hasattr(cfg, 'furniture')


def build_episodes(cfg):
    """-> list of {'observations', 'actions', 'rewards'} episode dicts."""
    if is_maze2d(cfg):
        from diffplan.tasks.maze2d import load_episodes, make_env

        return load_episodes(make_env(cfg.env_id))
    from diffplan.tasks.furniture_data import load_episodes

    return load_episodes(cfg.data_dir, n_success=cfg.n_success, n_failure=cfg.n_failure)


def build_dataset(cfg, episodes):
    kw = dict(horizon=cfg.horizon, cond_steps=tuple(cfg.cond_steps), discount=cfg.discount,
              normalizer=Normalizer.fit(episodes), pad_last=cfg.pad_last)
    if is_maze2d(cfg):
        return TrajectoryDataset(episodes, **kw)
    from diffplan.tasks.furniture_data import FurnitureDataset

    return FurnitureDataset(cfg.data_dir, episodes, **kw)


def build_models(cfg, observation_dim, action_dim):
    """-> (diffusion, value_model or None). Shared with scripts/eval.py, which takes the dims
    from the checkpoint's normalizer."""
    encoder = None if is_maze2d(cfg) else ImageEncoder(crop=cfg.crop, backbone=cfg.encoder)
    cond_dim = 0 if encoder is None else encoder.out_dim
    dim_mults = tuple(cfg.dim_mults)
    transition_dim = observation_dim + action_dim

    model = TemporalUnet(cfg.horizon, transition_dim, dim=cfg.dim, dim_mults=dim_mults, cond_dim=cond_dim)
    diffusion = GaussianDiffusion(
        model, horizon=cfg.horizon, transition_dim=transition_dim, action_dim=action_dim,
        n_timesteps=cfg.n_diffusion_steps, loss_type=cfg.loss_type, clip_denoised=cfg.clip_denoised,
        clip_value=cfg.clip, predict_epsilon=cfg.predict_epsilon, action_weight=cfg.action_weight,
        encoder=encoder)

    value_model = None
    if cfg.discount is not None:
        value_net = TemporalValue(cfg.horizon, transition_dim, dim=cfg.dim, dim_mults=dim_mults,
                                  action_dim=action_dim, cond_dim=cond_dim)
        value_model = ReturnPredictor(value_net, n_timesteps=cfg.n_diffusion_steps, action_dim=action_dim)
    return diffusion, value_model


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def build_probe(cfg, dataset, logdir, with_images=True):
    """-> probe(step, ema_diffusion) -> dict of metrics, or None if the task cannot be probed.

    Rolls out a few fixed-seed episodes with the EMA weights and returns the normalized score plus
    one plan picture per episode, because the loss curve cannot show the two things most likely to
    be wrong: a plan that clips a wall, or an inverted normalizer.

    None for furniture: probing it means stepping the simulator, i.e. importing isaacgym into the
    training process, which the note at the top of this file forbids.
    """
    if not is_maze2d(cfg):
        return None

    from diffplan.tasks.maze2d import make_env, plot_plan, rollout

    # One env, built once: building it per probe would leak a mujoco context per call.
    env = make_env(cfg.env_id)
    n_episodes = max(1, int(cfg.probe_episodes))
    plotdir = os.path.join(logdir, 'plans')

    def probe(step, ema):
        scores, paths = [], []
        for e in range(n_episodes):
            # Seeded per episode INDEX, not per call: the start state and the sampler noise are
            # then identical from one probe to the next, so the only thing that moved between two
            # readings is the weights. The trainer restores the RNG state afterwards.
            torch.manual_seed(e)
            np.random.seed(e)
            result = rollout(env, ema, dataset.normalizer, cfg, seed=e)
            scores.append(result['score'])
            paths.append(plot_plan(
                env, result['plan'], os.path.join(plotdir, 'step%08d_ep%d.png' % (step, e))))

        # x100 to match the paper's table, the same scaling scripts/eval.py uses -- but the probe
        # averages only probe_episodes of them, so it estimates the level rather than replacing
        # the 100-episode eval. Episode length is deliberately not reported: d4rl's MazeEnv.step
        # hardcodes done=False, so it is a dead constant, and the real speed information is
        # already in the score (maze2d pays +1 per step spent within 0.5 of the target).
        metrics = {'probe/normalized_score': 100.0 * float(np.mean(scores)),
                   'probe/score_std': 100.0 * float(np.std(scores))}
        if with_images:
            import wandb
            metrics['probe/plans'] = [wandb.Image(p, caption='ep %d' % e) for e, p in enumerate(paths)]
        return metrics

    return probe


def init_wandb(cfg, logdir, dataset):
    """-> a wandb Run, or None if logging is off or unreachable.

    Imported lazily so `--use_wandb false` runs without wandb installed, and fail-soft because
    losing a multi-day run to a transient network error at startup would be absurd.
    """
    if not cfg.use_wandb:
        return None
    try:
        import wandb

        config = config_to_dict(cfg)
        config.update({
            'n_windows': len(dataset),
            'observation_dim': dataset.observation_dim,
            'action_dim': dataset.action_dim,
            'transition_dim': dataset.transition_dim,
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'hostname': os.environ.get('SLURMD_NODENAME'),
        })
        return wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            # A readable name and a group, but deliberately NOT a fixed id with resume='allow':
            # resuming an id keeps wandb's old step counter, and a re-submitted job restarts from
            # step 0 (the checkpoint carries no optimizer state), so every metric would be
            # rejected for going backwards. A requeue therefore shows up as a second run in the
            # same group, which is at least honest.
            name='%s/%s' % (os.path.basename(cfg.logbase.rstrip('/')), run_name(cfg).replace('/', '__')),
            group=cfg.env_id,
            config=config,
            dir=logdir,
        )
    except Exception as e:
        print('[ train ] wandb disabled: %s: %s' % (type(e).__name__, e))
        return None


def main(argv=None):
    cfg = load_config(argv)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Worth 3.2x on an RTX 4090, measured: without it cuDNN picks convolution algorithms by
    # heuristic, and it picks badly for a TemporalUnet, whose deepest stage is short and wide.
    # Autotuning is the right trade only because our shapes never vary (drop_last=True and a
    # fixed horizon), so the one-time search is paid once rather than per new shape.
    torch.backends.cudnn.benchmark = True

    # --smoke shrinks the model in load_config but says nothing about hardware, so it may fall
    # back to CPU on a GPU-less login node. NOTHING ELSE MAY: a silent CPU fallback once turned a
    # 12 h run into a 250 h one that looked healthy in every other respect (torch's CUDA init had
    # failed with "CUDA unknown error", usually a missing nvidia_uvm module, while nvidia-smi saw
    # the GPU fine). A broken node must fail the job in the first second instead.
    device = cfg.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        assert cfg.smoke, (
            'cfg.device=%r but torch.cuda.is_available() is False. Refusing to train on CPU: '
            'it is ~20x slower and would silently burn the whole allocation. Check the node '
            '(nvidia-smi; whether nvidia_uvm is loaded), resubmit elsewhere with '
            '--exclude=<node>, or pass --device cpu to mean it.' % device)
        print('[ train ] cuda unavailable, falling back to cpu (--smoke)')
        device = 'cpu'

    episodes = build_episodes(cfg)
    dataset = build_dataset(cfg, episodes)
    diffusion, value_model = build_models(cfg, dataset.observation_dim, dataset.action_dim)

    logdir = os.path.join(cfg.logbase, run_name(cfg))
    os.makedirs(logdir, exist_ok=True)  # wandb wants its `dir` to exist already
    print('[ train ] %s' % logdir)
    print('[ train ] %d episodes | %d windows | obs %d | act %d | transition %d | horizon %d'
          % (len(episodes), len(dataset), dataset.observation_dim, dataset.action_dim,
             dataset.transition_dim, cfg.horizon))
    print('[ train ] unet params: %.2fM' % (count_params(diffusion.model) / 1e6))
    if value_model is not None:
        print('[ train ] value params: %.2fM' % (count_params(value_model) / 1e6))

    run = init_wandb(cfg, logdir, dataset)
    if run is not None:
        print('[ train ] wandb: %s' % run.url)
    # No images to build when there is no run to send them to; the pngs still land in logdir.
    probe = build_probe(cfg, dataset, logdir, with_images=run is not None)

    trainer = Trainer(
        diffusion, dataset, value_model=value_model, batch_size=cfg.batch_size, lr=cfg.lr,
        grad_accum=cfg.grad_accum, ema_decay=cfg.ema_decay, ema_start=cfg.ema_start,
        ema_every=cfg.ema_every, log_every=cfg.log_every, save_every=cfg.save_every,
        snapshot_every=cfg.snapshot_every, logdir=logdir, device=device,
        num_workers=cfg.num_workers,
        # The config travels inside the checkpoint: eval reconstructs the architecture from it
        # instead of re-deriving it from flags, so the two can never disagree.
        config=cfg, run=run, probe=probe, probe_every_seconds=cfg.probe_every_seconds,
        n_steps=int(cfg.n_train_steps), cosine_lr=cfg.cosine_lr)

    exit_code = 1
    try:
        trainer.train(int(cfg.n_train_steps))
        exit_code = 0
    finally:
        # Flush wandb even when training dies, so a crash still leaves the metrics that led up to
        # it. exit_code is EXPLICIT because finish() runs while the exception is still unwinding,
        # before wandb's own exit-code hook fires: it would otherwise record 0 and mark a run that
        # died at step 700k as green "Finished". This does not cover a slurm SIGTERM, where python
        # exits without unwinding; the loss there is at most log_every steps of curve.
        if run is not None:
            run.finish(exit_code=exit_code)
    return trainer


if __name__ == '__main__':
    main()
