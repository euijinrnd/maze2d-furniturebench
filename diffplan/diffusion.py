"""Gaussian diffusion over trajectories, plus a noise-aware return predictor for guidance.

Channel order in every trajectory tensor is [actions | observations]. Reference: Janner et al.,
"Planning with Diffusion for Flexible Behavior Synthesis" (arXiv:2205.09991).

`ctx` is an optional global context vector fed to the networks (image features for furniture),
produced by `GaussianDiffusion.encode(images)` and shared with the value model.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def cosine_beta_schedule(n_timesteps, s=0.008):
    """All diffusion constants as one dict of float32 buffers (Nichol & Dhariwal, upstream's off-by-one kept)."""
    steps = n_timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = torch.tensor(np.clip(1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), 0, 0.999), dtype=torch.float32)

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return {
        'betas': betas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1.0 - alphas_cumprod),
        'sqrt_recip_alphas_cumprod': torch.sqrt(1.0 / alphas_cumprod),
        'sqrt_recipm1_alphas_cumprod': torch.sqrt(1.0 / alphas_cumprod - 1),
        'posterior_variance': posterior_variance,
        'posterior_log_variance_clipped': torch.log(torch.clamp(posterior_variance, min=1e-20)),
        'posterior_mean_coef1': betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        'posterior_mean_coef2': (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
    }


def extract(a, t, x_shape):
    return a.gather(-1, t).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def apply_conditioning(x, cond, action_dim):
    """Inpaint observations at the conditioned timesteps. Mutates and returns x."""
    for h, obs in cond.items():
        x[:, h, action_dim:action_dim + obs.shape[-1]] = obs
    return x


def _pearson(pred, targ):
    p = pred.detach().float().reshape(-1)
    t = targ.detach().float().reshape(-1)
    p, t = p - p.mean(), t - t.mean()
    denom = p.norm() * t.norm()
    return p @ t / denom if p.numel() > 1 and denom.item() > 1e-12 else torch.full((), float('nan'), device=pred.device)


class GaussianDiffusion(nn.Module):
    """DDPM over fixed-length trajectories with observation inpainting."""

    def __init__(self, model, horizon, transition_dim, action_dim, n_timesteps=20, loss_type='l2',
                 clip_denoised=True, clip_value=1.0, predict_epsilon=False, action_weight=1.0, encoder=None):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.horizon = int(horizon)
        self.transition_dim = int(transition_dim)
        self.action_dim = int(action_dim)
        self.n_timesteps = int(n_timesteps)
        self.loss_type = loss_type
        self.clip_denoised = clip_denoised
        self.clip_value = float(clip_value)
        self.predict_epsilon = predict_epsilon
        for name, buf in cosine_beta_schedule(self.n_timesteps).items():
            self.register_buffer(name, buf)
        # All ones, with a_0 upweighted: it is the only channel executed on the very next step.
        loss_weights = torch.ones(self.horizon, self.transition_dim)
        loss_weights[0, :self.action_dim] = action_weight
        self.register_buffer('loss_weights', loss_weights)

    def encode(self, images):
        return None if images is None else self.encoder(images)

    # ---------------------------- sampling ---------------------------- #

    def predict_start_from_noise(self, x_t, t, out):
        if not self.predict_epsilon:
            return out
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * out)

    def q_posterior(self, x_start, x_t, t):
        mean = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        return mean, extract(self.posterior_log_variance_clipped, t, x_t.shape)

    @torch.no_grad()
    def p_sample(self, x, cond, t, ctx, guide=None, guide_scale=0.1, t_stopgrad=2, n_guide_steps=1):
        """One reverse step, optionally preceded by n_guide_steps of value-gradient ascent."""
        if guide is not None:
            var = extract(self.posterior_variance, t, x.shape)
            for _ in range(n_guide_steps):
                _, grad = guide.gradients(x, t, ctx)
                grad = var * grad
                grad[t < t_stopgrad] = 0  # nearly clean: pushing further only adds off-manifold jitter
                x = apply_conditioning(x + guide_scale * grad, cond, self.action_dim)

        x_recon = self.predict_start_from_noise(x, t, self.model(x, t, ctx))
        if self.clip_denoised:
            x_recon = x_recon.clamp(-self.clip_value, self.clip_value)
        mean, log_var = self.q_posterior(x_recon, x, t)
        noise = torch.randn_like(x)
        noise[t == 0] = 0
        return mean + (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def sample(self, cond, batch_size=1, ctx=None, guide=None, **guide_kwargs):
        """Run the full reverse chain. Returns [batch_size, horizon, transition_dim]."""
        device = self.betas.device
        x = apply_conditioning(torch.randn(batch_size, self.horizon, self.transition_dim, device=device), cond, self.action_dim)
        for i in reversed(range(self.n_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, t, ctx, guide=guide, **guide_kwargs)
            x = apply_conditioning(x, cond, self.action_dim)  # re-pin: inpainting IS the planning mechanism
        return x

    # ---------------------------- training ---------------------------- #

    def q_sample(self, x_start, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def loss(self, x, cond, ctx=None):
        t = torch.randint(0, self.n_timesteps, (len(x),), device=x.device).long()
        noise = torch.randn_like(x)
        x_noisy = apply_conditioning(self.q_sample(x, t, noise), cond, self.action_dim)
        x_recon = self.model(x_noisy, t, ctx)
        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, self.action_dim)
        target = noise if self.predict_epsilon else x
        elem = F.mse_loss(x_recon, target, reduction='none') if self.loss_type == 'l2' else (x_recon - target).abs()
        loss = (elem * self.loss_weights).mean()
        info = {'loss': loss.detach()}
        if self.action_dim:  # a state-only high level has no a_0 to report
            info['a0_loss'] = elem[:, 0, :self.action_dim].mean().detach()
        return loss, info


class ReturnPredictor(nn.Module):
    """Noise-aware value net used as the guidance signal; shares the diffusion schedule bit for bit."""

    def __init__(self, model, n_timesteps, action_dim):
        super().__init__()
        self.model = model
        self.n_timesteps = int(n_timesteps)
        self.action_dim = int(action_dim)
        for name, buf in cosine_beta_schedule(self.n_timesteps).items():
            self.register_buffer(name, buf)

    def forward(self, x, t, ctx=None):
        return self.model(x, t, ctx)

    def gradients(self, x, t, ctx=None):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            y = self(x, t, ctx)
            grad = torch.autograd.grad([y.sum()], [x])[0]
        return y.detach(), grad.detach()

    def loss(self, x, cond, value, ctx=None):
        """Regress the discounted return from a NOISED trajectory (Dhariwal & Nichol's noise-aware classifier)."""
        t = torch.randint(0, self.n_timesteps, (len(x),), device=x.device).long()
        noise = torch.randn_like(x)
        x_noisy = (extract(self.sqrt_alphas_cumprod, t, x.shape) * x
                   + extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise)
        pred = self(apply_conditioning(x_noisy, cond, self.action_dim), t, ctx)
        target = value.to(pred.dtype).reshape(-1)
        loss = F.mse_loss(pred, target)
        return loss, {'value_loss': loss.detach(), 'value_corr': _pearson(pred, target)}


@torch.no_grad()
def plan(diffusion, normalizer, cond, batch_size=1, images=None, guide=None, device='cuda', **guide_kwargs):
    """Normalize -> sample -> (optionally) rank -> unnormalize.

    cond maps timestep -> np.ndarray[obs_dim] in RAW units, or [batch_size, obs_dim] to pin a
    different value per sample (plan_hd gives every row its own pair of subgoals); images is the
    raw uint8 [n_cams, H, W, 3] frame the plan is conditioned on. Returns (actions, observations)
    as raw-unit numpy arrays [batch_size, H, dim], with row 0 the best plan when a guide is given.
    """
    cond_t = {}
    for h, obs in cond.items():
        obs = torch.as_tensor(normalizer.normalize(np.asarray(obs, dtype=np.float32), 'observations'), device=device)
        cond_t[int(h)] = obs.reshape(-1, obs.shape[-1]).expand(batch_size, -1)
    ctx = None
    if images is not None:
        ctx = diffusion.encode(torch.as_tensor(np.asarray(images), device=device)[None]).expand(batch_size, -1)

    x = diffusion.sample(cond_t, batch_size=batch_size, ctx=ctx, guide=guide, **guide_kwargs)
    if guide is not None:
        values = guide(x, torch.zeros(batch_size, device=x.device, dtype=torch.long), ctx)
        x = x[torch.argsort(values, descending=True)]

    d = diffusion.action_dim
    # A state-only high-level model has d == 0: there is nothing to unnormalize, and the
    # normalizer's action statistics would broadcast against the empty slice rather than stay empty.
    actions = normalizer.unnormalize_torch(x[:, :, :d], 'actions') if d else x[:, :, :d]
    return (actions.cpu().numpy(),
            normalizer.unnormalize_torch(x[:, :, d:], 'observations').cpu().numpy())


@torch.no_grad()
def plan_hd(high, low, normalizer, cond, jump, images=None, n_segments=None, device='cuda'):
    """Hierarchical Diffuser planning: sparse subgoals, then one dense segment per gap.

    `cond` conditions the HIGH level in raw units -- {0: current state}, plus {n_subgoal - 1: goal}
    for maze2d. The low level then fills every gap with both endpoints inpainted, all segments in
    ONE batch (Chen et al., 2024, Alg. 1-3). Returns the stitched dense (actions, observations) of
    length jump * n_segments + 1, with the same leading batch axis of 1 that plan() gives, so the
    rollouts keep indexing [0].

    `n_segments=1` is the closed-loop branch of Alg. 3: only the first segment is needed when we
    replan after a few steps, and it is the only one whose start matches the frame in `images`.

    No guidance: maze2d trains no value model, one_leg's guide costs success rate (docs/one_leg.md),
    and plan()'s value ranking would reorder a batch whose rows are consecutive segments.
    """
    _, subgoals = plan(high, normalizer, cond, images=images, device=device)
    g = subgoals[0]  # [n_subgoal, observation_dim], raw units
    n = len(g) - 1 if n_segments is None else int(n_segments)
    assert 1 <= n < len(g), 'asked for %d segments from %d subgoals' % (n, len(g))
    assert images is None or n == 1, 'only the first segment starts at the conditioning frame'

    actions, observations = plan(low, normalizer, {0: g[:n], jump: g[1:n + 1]},
                                 batch_size=n, images=images, device=device)

    def stitch(z):
        # Segment i ends where segment i + 1 begins: [n, jump + 1, d] -> [1, n * jump + 1, d].
        return np.concatenate(list(z[:, :-1]) + [z[-1, -1:]])[None]

    return stitch(actions), stitch(observations)
