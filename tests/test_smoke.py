"""CPU-only invariants: channel layout, windows, normalizer, rotations, conditioning, a tiny overfit."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from diffplan.config import CONFIGS, load_config, run_name
from diffplan.dataset import Normalizer, TrajectoryDataset
from diffplan.diffusion import GaussianDiffusion, ReturnPredictor, apply_conditioning, cosine_beta_schedule, plan
from diffplan.nn import ImageEncoder, TemporalUnet, TemporalValue
from diffplan.tasks import furniture_data as fd


def ramp_episodes(n_ep=4, length=24, step=0.1):
    """Straight-line 2-d episodes moving along +x with constant actions."""
    eps = []
    for i in range(n_ep):
        obs = np.array([[-1.0, 0.25 * i]]) + np.arange(length)[:, None] * np.array([[step, 0.0]])
        eps.append({'observations': obs.astype(np.float32),
                    'actions': np.tile([[step, 0.0]], (length, 1)).astype(np.float32),
                    'rewards': np.eye(length, dtype=np.float32)[-1]})
    return eps


def test_normalizer_round_trip_and_state_dict():
    eps = ramp_episodes()
    eps[0]['observations'][:, 1] = 3.0  # a constant dimension must not divide by zero
    n = Normalizer.fit(eps)
    x = eps[1]['observations']
    assert np.allclose(n.unnormalize(n.normalize(x, 'observations'), 'observations'), x, atol=1e-5)
    m = Normalizer.from_state_dict(n.state_dict())
    assert np.allclose(m.normalize(x, 'observations'), n.normalize(x, 'observations'))
    t = torch.as_tensor(n.normalize(x, 'observations'))
    assert np.array_equal(m.unnormalize_torch(t, 'observations').numpy(), n.unnormalize(t.numpy(), 'observations'))


def test_unnormalize_torch_is_bitwise_identical():
    """plan() runs the torch path; anything short of exact equality makes evals incomparable."""
    rng = np.random.default_rng(0)
    dim = 137
    loc = rng.normal(0, 1, dim).astype(np.float32)
    scale = np.maximum(np.abs(rng.normal(0, 1, dim)), 1e-6).astype(np.float32)
    norm = Normalizer({'observations': loc}, {'observations': scale})
    x = rng.normal(0, 2.5, (16, 8, dim)).astype(np.float32)
    x[0, :5] = np.array([1.0, -1.0, 1.5, -1.5, np.nextafter(np.float32(1), np.float32(0))])[:, None]
    want = norm.unnormalize(x, 'observations')
    got = norm.unnormalize_torch(torch.from_numpy(x), 'observations').numpy()
    assert got.dtype == want.dtype == np.float32
    assert np.array_equal(got, want), 'max abs diff %g' % np.abs(got - want).max()


def test_dataset_windows_values_and_padding():
    eps = ramp_episodes(n_ep=2, length=10)
    ds = TrajectoryDataset(eps, horizon=4, cond_steps=(0, -1), discount=0.5)
    assert len(ds) == 2 * (10 - 4 + 1) and ds.transition_dim == 4 and set(ds.cond_steps) == {0, 3}
    b = ds[6]  # the last window of episode 0 ends on the reward frame
    assert b['x'].shape == (4, 4) and np.isclose(b['value'], 1.0)  # max return-to-go -> +1
    assert np.allclose(b['cond'][3], b['x'][3, 2:])
    padded = TrajectoryDataset(eps, horizon=4, pad_last=True)
    assert len(padded) == 20 and np.allclose(padded[9]['x'][1:], padded[9]['x'][:1])
    batch = next(iter(DataLoader(ds, batch_size=3)))
    assert batch['x'].shape == (3, 4, 4) and batch['cond'][0].shape == (3, 2)


def test_strided_windows_are_the_high_level_view():
    """The high level sees every K-th state of a window, with the actions dropped (eq. 11)."""
    eps = ramp_episodes(n_ep=2, length=10)
    ds = TrajectoryDataset(eps, horizon=4, stride=3, include_actions=False, cond_steps=(0, -1))
    # A window now spans (4 - 1) * 3 + 1 = 10 transitions, so exactly one start per episode.
    assert len(ds) == 2 and ds.action_dim == 0 and ds.transition_dim == ds.observation_dim
    assert np.allclose(ds[0]['x'], ds.normalizer.normalize(eps[0]['observations'][::3], 'observations'))
    assert np.allclose(ds[0]['cond'][3], ds[0]['x'][3])
    # Padding still clamps to the final transition, stride or no stride.
    padded = TrajectoryDataset(eps, horizon=4, stride=3, pad_last=True)
    assert len(padded) == 20 and np.allclose(padded[9]['x'][-1], padded[9]['x'][-2])


def test_apply_conditioning_only_touches_observations():
    x = torch.zeros(2, 5, 4)
    out = apply_conditioning(x, {0: torch.ones(2, 2), 4: 2 * torch.ones(2, 2)}, action_dim=2)
    assert out[:, 0, 2:].eq(1).all() and out[:, 4, 2:].eq(2).all() and out[:, :, :2].eq(0).all()
    with pytest.raises(RuntimeError):
        apply_conditioning(torch.zeros(2, 5, 4), {0: torch.ones(2, 3)}, action_dim=2)


def test_apply_conditioning_with_no_action_channels():
    """The high level has action_dim 0, so conditioning starts at channel 0."""
    out = apply_conditioning(torch.zeros(2, 5, 3), {0: torch.ones(2, 3)}, action_dim=0)
    assert out[:, 0].eq(1).all() and out[:, 1:].eq(0).all()


def test_schedule_constants():
    s = cosine_beta_schedule(20)
    assert s['betas'].shape == (20,) and (s['betas'] > 0).all() and (s['betas'] <= 0.999).all()
    assert torch.allclose(s['alphas_cumprod'], torch.cumprod(1 - s['betas'], 0))


@pytest.mark.parametrize('cond_dim', [0, 8])
def test_networks_shapes_and_value_mask(cond_dim):
    unet = TemporalUnet(16, 6, dim=16, dim_mults=(1, 2, 4), cond_dim=cond_dim)
    value = TemporalValue(16, 6, dim=16, dim_mults=(1, 2, 4), action_dim=2, cond_dim=cond_dim)
    x, t = torch.randn(3, 16, 6), torch.randint(0, 20, (3,))
    ctx = torch.randn(3, cond_dim) if cond_dim else None
    assert unet(x, t, ctx).shape == (3, 16, 6) and value(x, t, ctx).shape == (3,)
    y = x.clone()
    y[:, 0, 2:] = 5.0  # conditioned observation at step 0 is masked ...
    assert torch.allclose(value(x, t, ctx), value(y, t, ctx))
    y[:, 0, :2] = 5.0  # ... but a_0 is not
    assert not torch.allclose(value(x, t, ctx), value(y, t, ctx))
    with pytest.raises(AssertionError):
        TemporalUnet(18, 6, dim=16, dim_mults=(1, 2, 4))


def test_image_encoder_crops():
    enc = ImageEncoder(n_cams=2, crop=32)
    imgs = torch.randint(0, 255, (2, 2, 40, 40, 3), dtype=torch.uint8)
    assert enc(imgs).shape == (2, enc.out_dim)
    enc.eval()
    assert enc(imgs).shape == (2, enc.out_dim)


def test_rotations_and_action_round_trip():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(50, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    m = fd.quat_to_mat(q)
    assert np.allclose(np.abs((fd.mat_to_quat(m) * q).sum(1)), 1, atol=1e-5)
    assert np.allclose(fd.rot6d_to_mat(fd.mat_to_rot6d(m)), m, atol=1e-6)
    assert np.allclose(fd.quat_to_mat(fd.quat_mul(q[:10], q[10:20])), m[:10] @ m[10:20], atol=1e-5)
    s = np.concatenate([rng.normal(size=(10, 3)), q[:10], rng.normal(size=(10, 7))], 1).astype(np.float32)
    d = np.concatenate([rng.normal(size=(10, 3)) * 0.05, q[20:30], rng.choice([-1.0, 1.0], size=(10, 1))], 1).astype(np.float32)
    back = np.stack([fd.decode_action(t, si) for t, si in zip(fd.encode_action(s, d), s)])
    assert np.allclose(back[:, :3], d[:, :3], atol=1e-5) and np.allclose(back[:, 7], d[:, 7])
    assert np.allclose(np.abs((back[:, 3:7] * d[:, 3:7]).sum(1)), 1, atol=1e-5)
    assert fd.encode_state(s).shape == (10, fd.OBS_DIM) and fd.encode_action(s, d).shape == (10, fd.ACT_DIM)


def test_configs():
    assert (CONFIGS['maze2d-large'].horizon, CONFIGS['maze2d-large'].n_diffusion_steps) == (384, 256)
    one_leg = CONFIGS['one_leg']
    # The reported one_leg setting: success-only demos, a single unguided sample, replan every 2.
    assert (one_leg.horizon, one_leg.n_diffusion_steps, one_leg.discount) == (32, 20, 0.997)
    assert (one_leg.n_failure, one_leg.plan_batch_size, one_leg.replan_every) == (0, 1, 2)
    cfg = load_config(['--config', 'one_leg', '--smoke', '--horizon', '16'])
    assert cfg.smoke and cfg.horizon == 16 and cfg.device == 'cpu' and run_name(cfg) == 'one_leg/H16_T4_s0'


def test_overfit_reproduces_conditioned_endpoints_and_guidance_runs():
    torch.manual_seed(0)
    np.random.seed(0)
    horizon, eps = 8, ramp_episodes()
    ds = TrajectoryDataset(eps, horizon=horizon, cond_steps=(0, -1), discount=0.9)
    unet = TemporalUnet(horizon, ds.transition_dim, dim=16, dim_mults=(1, 2, 4))
    diffusion = GaussianDiffusion(unet, horizon, ds.transition_dim, ds.action_dim, n_timesteps=20)
    guide = ReturnPredictor(TemporalValue(horizon, ds.transition_dim, dim=16, dim_mults=(1, 2, 4), action_dim=2), 20, 2)
    opt = torch.optim.Adam(list(diffusion.parameters()) + list(guide.parameters()), lr=1e-3)
    losses = []
    for step, batch in enumerate(iter(lambda: next(iter(DataLoader(ds, batch_size=32, shuffle=True))), None)):
        loss, _ = diffusion.loss(batch['x'], batch['cond'])
        v_loss, v_info = guide.loss(batch['x'], batch['cond'], batch['value'])
        opt.zero_grad()
        (loss + v_loss).backward()
        opt.step()
        losses.append(float(loss))
        if step == 400:
            break
    assert np.mean(losses[-20:]) < 0.5 * np.mean(losses[:20]) and np.isfinite(float(v_info['value_corr']))

    diffusion.eval()
    start, goal = eps[0]['observations'][3], eps[0]['observations'][3 + horizon - 1]
    actions, obs = plan(diffusion, ds.normalizer, {0: start, horizon - 1: goal}, batch_size=4, device='cpu',
                        guide=guide, guide_scale=0.1, t_stopgrad=2, n_guide_steps=2)
    assert actions.shape == (4, horizon, 2) and obs.shape == (4, horizon, 2)
    assert np.allclose(obs[:, 0], start, atol=1e-3) and np.allclose(obs[:, -1], goal, atol=1e-3)
    assert actions[0, :-1, 0].mean() > 0.04 and abs(actions[0, :-1, 1].mean()) < 0.06
