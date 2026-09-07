"""Training loop for the trajectory diffusion model (and optionally the return predictor).

One loop, one optimizer per model, one self-contained checkpoint.
"""
from __future__ import annotations

import copy
import json
import os
import time
import traceback

import numpy as np
import torch
from tqdm import tqdm


def cycle(dataloader):
    """Infinite iterator: Diffuser counts optimizer steps, not epochs."""
    while True:
        for batch in dataloader:
            yield batch


@torch.no_grad()
def ema_update(ema_model, model, decay):
    """In-place EMA of parameters; buffers are hard-copied.

    Upstream EMAs only `.parameters()` and leaves the EMA model's buffers frozen at whatever the
    initial deepcopy captured. That is harmless here (every buffer is a schedule constant) but is
    a trap the moment anyone adds a BatchNorm or a step counter.
    """
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)


def _to_device(batch, device):
    """Move a batch dict to `device`, recursing into the int-keyed `cond` dict."""
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    return batch


class Trainer:
    def __init__(self, diffusion, dataset, value_model=None, batch_size=32, lr=2e-4,
                 grad_accum=2, ema_decay=0.995, ema_start=2000, ema_every=10,
                 log_every=100, save_every=10000, snapshot_every=100000,
                 logdir='logs/run', device='cuda', num_workers=1, config=None,
                 run=None, probe=None, probe_every_seconds=3600.0, n_steps=None, cosine_lr=False):
        self.diffusion = diffusion.to(device)
        self.value_model = value_model.to(device) if value_model is not None else None
        self.dataset = dataset
        self.device = device
        self.config = config

        # The EMA weights are what we plan with. deepcopy clones every parameter and buffer
        # on-device, so it is the whole initial sync. The copies are only ever planned with or
        # serialized, never trained, so eval() is their resting state.
        self.ema = copy.deepcopy(self.diffusion).requires_grad_(False).eval()
        self.value_ema = None
        if self.value_model is not None:
            self.value_ema = copy.deepcopy(self.value_model).requires_grad_(False).eval()

        # Plain Adam, no clipping and no AMP: the paper trains exactly like this, and every
        # deviation is a confound when comparing against published numbers.
        self.optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, n_steps) if cosine_lr else None
        self.value_optimizer = None
        if self.value_model is not None:
            self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=lr)

        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_every = ema_every
        self.log_every = log_every
        self.save_every = save_every
        self.snapshot_every = snapshot_every
        self.logdir = logdir
        os.makedirs(self.logdir, exist_ok=True)

        # `run` is duck-typed: anything with .log(dict, step=int), in practice a wandb Run.
        # Nothing in diffplan/ imports wandb, so the trainer stays testable by passing None.
        self.run = run
        # `probe(step, ema_model) -> dict` is a task-specific mid-training evaluation, injected
        # rather than implemented here because probing maze2d means importing d4rl. Values may be
        # numbers or wandb.Image objects; the numbers also go to the console and probe.log.
        self.probe = probe
        self.probe_every_seconds = float(probe_every_seconds)

        # With drop_last=True a dataset smaller than one batch yields ZERO batches and cycle()
        # would spin forever printing nothing.
        assert len(dataset) >= batch_size, (
            'dataset has %d windows < batch_size %d: with drop_last=True the dataloader '
            'would never yield a batch' % (len(dataset), batch_size))

        self.dataloader = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=str(device).startswith('cuda'), drop_last=True))
        self.step = 0

    def train(self, n_steps):
        self.diffusion.train()
        if self.value_model is not None:
            self.value_model.train()

        start = time.time()
        infos = {}
        # 0.0, so the first check fires a probe immediately: that proves the whole rollout ->
        # plot -> upload path works within the first minute. The step-0 plan comes from an
        # untrained model and is expected to look like noise -- that image IS the baseline.
        last_probe = 0.0
        progress = tqdm(range(n_steps), ncols=100, desc=os.path.basename(self.logdir))
        for _ in progress:
            self.optimizer.zero_grad()
            if self.value_optimizer is not None:
                self.value_optimizer.zero_grad()

            for _ in range(self.grad_accum):
                batch = _to_device(next(self.dataloader), self.device)
                ctx = self.diffusion.encode(batch.get('images'))

                loss, info = self.diffusion.loss(batch['x'], batch['cond'], ctx)
                (loss / self.grad_accum).backward()
                infos = dict(info)

                if self.value_model is not None:
                    # ctx is detached: the value loss must not train the shared image encoder.
                    v_loss, v_info = self.value_model.loss(
                        batch['x'], batch['cond'], batch['value'], None if ctx is None else ctx.detach())
                    (v_loss / self.grad_accum).backward()
                    infos.update(v_info)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.value_optimizer is not None:
                # Only the value gradients are clipped: its head diverges at dim >= 128 (observed
                # value_loss 7e5 against targets in [-1, 0]) while the diffusion model trains fine
                # on the same Adam settings.
                torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1.0)
                self.value_optimizer.step()

            if self.step % self.ema_every == 0:
                # Before `ema_start` the weights move too fast for an EMA to mean anything, so
                # hard-reset instead of averaging (upstream's reset_parameters).
                if self.step < self.ema_start:
                    self.ema.load_state_dict(self.diffusion.state_dict())
                    if self.value_ema is not None:
                        self.value_ema.load_state_dict(self.value_model.state_dict())
                else:
                    ema_update(self.ema, self.diffusion, self.ema_decay)
                    if self.value_ema is not None:
                        ema_update(self.value_ema, self.value_model, self.ema_decay)

            if self.step % self.log_every == 0:
                # `loss` is the last micro-batch's UNDIVIDED loss. Upstream logs it after dividing
                # by the accumulation count, so its printed losses are grad_accum times smaller
                # than the actual objective.
                line = ' | '.join('{}: {:8.5f}'.format(k, float(v)) for k, v in infos.items())
                progress.set_postfix_str(line[:60])
                tqdm.write('{:>8}: {} | {:6.1f}s'.format(self.step, line, time.time() - start))
                row = {k: float(v) for k, v in infos.items()}
                row['loss'] = float(loss)
                # `step` goes in the jsonl line but NOT in the remote payload: log_remote passes it
                # as the step= kwarg, and repeating it would create a series duplicating the x-axis.
                with open(os.path.join(self.logdir, 'train.log'), 'a') as f:
                    f.write(json.dumps(dict(row, step=self.step)) + '\n')
                self.log_remote(row)

            self.step += 1

            if self.step % self.save_every == 0 or self.step == n_steps:
                # A stable filename, rewritten in place, so eval and slurm arrays never glob for
                # the newest epoch. Numbered snapshots are archival and far rarer: at ~1 GB each
                # for one_leg, one per save would fill the disk.
                self.save(os.path.join(self.logdir, 'state_latest.pt'))
                if self.step % self.snapshot_every == 0 or self.step == n_steps:
                    self.save(os.path.join(self.logdir, 'state_{}.pt'.format(self.step)))

            # After the save block, so the probe's step is always >= the last logged one: wandb
            # rejects a log at a step lower than one it has seen, and the loss block above logs
            # the PRE-increment step while this logs the post-increment one.
            if self.probe is not None and time.time() - last_probe >= self.probe_every_seconds:
                last_probe = time.time()
                self.run_probe()

        progress.close()

    def log_remote(self, row):
        """Forward a row to `run`, if there is one. Never fatal: a dropped metric is not worth
        losing a multi-day run over, and wandb's transport can fail transiently."""
        if self.run is None:
            return
        try:
            self.run.log(row, step=self.step)
        except Exception:
            tqdm.write('[ trainer ] remote log failed at step %d:\n%s'
                       % (self.step, traceback.format_exc()))

    def run_probe(self):
        """Run the injected probe on the EMA weights and log whatever it returns.

        RNG state is saved and restored around the call: the probe seeds the global generators so
        its score is comparable from one hour to the next, and without the restore that seeding
        would reset the stream the diffusion loss draws `t` and `noise` from -- merely observing
        the run would change it. Exceptions are caught because a diagnostic that raises at hour 30
        of a three-day job must not take the training with it.
        """
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        np_rng = np.random.get_state()
        try:
            metrics = self.probe(self.step, self.ema)
        except Exception:
            tqdm.write('[ trainer ] probe failed at step %d (training continues):\n%s'
                       % (self.step, traceback.format_exc()))
            metrics = None
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            np.random.set_state(np_rng)

        if not metrics:
            return
        # Numbers go everywhere; richer objects (wandb.Image) only to `run`.
        scalars = {k: float(v) for k, v in metrics.items()
                   if isinstance(v, (bool, int, float, np.integer, np.floating))}
        tqdm.write('[ probe ] step %d | %s' % (
            self.step, ' | '.join('%s %.4f' % kv for kv in sorted(scalars.items()))))
        with open(os.path.join(self.logdir, 'probe.log'), 'a') as f:
            f.write(json.dumps(dict(scalars, step=self.step)) + '\n')
        self.log_remote(metrics)

    def save(self, path):
        """Write ONE self-contained checkpoint: both models, both EMAs, the normalizer and the
        config. Eval therefore never rebuilds the dataset, and the value model cannot be paired
        with a different normalizer or diffusion model than the one it shipped with (upstream
        pairs them by string-formatted path, which mismatches whenever a hyperparameter differs).

        The config is stored as the dataclass INSTANCE: diffplan.config imports no torch, so
        unpickling it never drags in a simulator.
        """
        data = {
            'step': self.step,
            'model': self.diffusion.state_dict(),
            'ema': self.ema.state_dict(),
            'value': self.value_model.state_dict() if self.value_model is not None else None,
            'value_ema': self.value_ema.state_dict() if self.value_ema is not None else None,
            'normalizer': self.dataset.normalizer.state_dict(),
            'config': self.config,
        }
        # Write-then-rename: state_latest.pt is overwritten hundreds of times, and a kill landing
        # inside an in-place write would leave a truncated file that fails much later as a
        # confusing unpickling error. os.replace is atomic within a filesystem.
        tmp = path + '.tmp'
        torch.save(data, tmp)
        os.replace(tmp, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.step = data['step']
        self.diffusion.load_state_dict(data['model'])
        self.ema.load_state_dict(data['ema'])
        if self.value_model is not None and data.get('value') is not None:
            self.value_model.load_state_dict(data['value'])
            self.value_ema.load_state_dict(data['value_ema'])
        return data
