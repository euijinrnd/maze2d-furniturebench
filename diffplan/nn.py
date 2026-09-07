"""Temporal (1D-conv over the horizon) networks for Diffuser, plus the image encoder.

The x0 model is a U-Net over the time axis of a trajectory x : [batch, horizon, transition_dim]
with channel order [actions | observations]. Every network takes an optional global context
vector `ctx` (image features for furniture) that rides with the diffusion-time embedding into
every block.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torchvision


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -(math.log(10000) / (half_dim - 1)))
        emb = t[:, None] * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, inp, out, kernel_size=5, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp, out, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    """Two conv blocks with an additive per-channel shift from the embedding, plus a residual."""

    def __init__(self, inp, out, embed_dim, kernel_size=5):
        super().__init__()
        self.blocks = nn.ModuleList([Conv1dBlock(inp, out, kernel_size), Conv1dBlock(out, out, kernel_size)])
        self.time_mlp = nn.Sequential(nn.Mish(), nn.Linear(embed_dim, out))
        self.residual_conv = nn.Conv1d(inp, out, 1) if inp != out else nn.Identity()

    def forward(self, x, emb):
        out = self.blocks[0](x) + self.time_mlp(emb)[:, :, None]
        return self.blocks[1](out) + self.residual_conv(x)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


def _time_mlp(dim):
    return nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim))


def _embed(time_mlp, t, ctx):
    emb = time_mlp(t)
    return emb if ctx is None else torch.cat([emb, ctx], dim=-1)


class TemporalUnet(nn.Module):
    """U-Net over the horizon axis. Predicts x0 (or epsilon) for a whole trajectory at once."""

    def __init__(self, horizon, transition_dim, dim=32, dim_mults=(1, 4, 8), cond_dim=0):
        super().__init__()
        n_down = 2 ** (len(dim_mults) - 1)
        assert horizon % n_down == 0, 'horizon %d must be divisible by %d' % (horizon, n_down)

        dims = [transition_dim] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.time_mlp = _time_mlp(dim)
        embed_dim = dim + cond_dim

        # Upstream's ASYMMETRIC topology is kept on purpose: the down path has len(in_out) stages,
        # the up path only len(in_out) - 1, so the highest-resolution skip is pushed and never
        # popped. The published numbers come from exactly this network.
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        n = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim),
                Downsample1d(dim_out) if ind < n - 1 else nn.Identity(),
            ]))
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim),
                Upsample1d(dim_in) if ind < n - 1 else nn.Identity(),
            ]))
        self.final_conv = nn.Sequential(Conv1dBlock(dim, dim, kernel_size=5), nn.Conv1d(dim, transition_dim, 1))

    def forward(self, x, t, ctx=None):
        # x : [B, horizon, transition_dim], t : [B], ctx : [B, cond_dim] or None
        x = x.transpose(1, 2)
        emb = _embed(self.time_mlp, t, ctx)
        h = []
        for resnet, resnet2, downsample in self.downs:
            x = resnet2(resnet(x, emb), emb)
            h.append(x)
            x = downsample(x)
        x = self.mid_block2(self.mid_block1(x, emb), emb)
        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = upsample(resnet2(resnet(x, emb), emb))
        return self.final_conv(x).transpose(1, 2)


class TemporalValue(nn.Module):
    """Return predictor J(x_t, t): a pure downsampling trunk -> flatten -> MLP -> scalar."""

    def __init__(self, horizon, transition_dim, dim=32, dim_mults=(1, 4, 8), action_dim=0, cond_dim=0):
        super().__init__()
        dims = [transition_dim] + [dim * m for m in dim_mults]
        self.action_dim = action_dim
        self.time_mlp = _time_mlp(dim)
        embed_dim = dim + cond_dim
        self.blocks = nn.ModuleList([
            nn.ModuleList([ResidualTemporalBlock(dim_in, dim_out, embed_dim),
                           ResidualTemporalBlock(dim_out, dim_out, embed_dim),
                           Downsample1d(dim_out)])
            for dim_in, dim_out in zip(dims[:-1], dims[1:])
        ])
        # Upstream computes the flattened size arithmetically and gets it wrong for (1,4,8); measure it.
        with torch.no_grad():
            fc_dim = self._trunk(torch.zeros(1, transition_dim, horizon), torch.zeros(1, embed_dim)).numel()
        # LayerNorm bounds the head input: the un-normalized trunk otherwise drives Mish to a hard
        # float32 zero at dim >= 128, after which the net is a constant and guidance a no-op.
        self.head = nn.Sequential(nn.LayerNorm(fc_dim + embed_dim), nn.Linear(fc_dim + embed_dim, fc_dim // 2),
                                  nn.Mish(), nn.Linear(fc_dim // 2, 1))

    def _trunk(self, x, emb):
        for resnet, resnet2, downsample in self.blocks:
            x = downsample(resnet2(resnet(x, emb), emb))
        return x

    def forward(self, x, t, ctx=None):
        # The observation at step 0 is pinned by inpainting, identical across all candidate plans;
        # zero it so the value keys on what the sampler actually generates (a_0 included).
        x = x.transpose(1, 2).clone()
        x[:, self.action_dim:, 0] = 0
        emb = _embed(self.time_mlp, t, ctx)
        return self.head(torch.cat([self._trunk(x, emb).flatten(1), emb], dim=-1)).squeeze(-1)


# --------------------------------------------------------------------------- images

class SpatialSoftmax(nn.Module):
    """Feature map [B, C, H, W] -> n_kp keypoints -> expected (x, y) per keypoint, [B, 2 n_kp]."""

    def __init__(self, in_channels, n_kp=64):
        super().__init__()
        self.kp = nn.Conv2d(in_channels, n_kp, 1)

    def forward(self, f):
        f = self.kp(f)
        b, c, h, w = f.shape
        p = torch.softmax(f.flatten(2), dim=-1).view(b, c, h, w)
        ys = torch.linspace(-1, 1, h, device=f.device)
        xs = torch.linspace(-1, 1, w, device=f.device)
        return torch.cat([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], dim=-1)


class ImageEncoder(nn.Module):
    """Per-camera visual features -> one context vector. Input uint8 [B, n_cams, H, W, 3].

    backbone 'resnet18': GroupNorm ResNet18 trained from scratch per camera (the Diffusion Policy
    recipe). backbone 'dinov2': one frozen DINOv2 ViT-B/14 (HDFlow's encoder) shared by the cameras,
    with a trainable head per camera. Both pool the final feature map with a spatial softmax; dinov2
    additionally projects its CLS token. Training applies a random crop of `crop` pixels, eval a center crop.
    """

    def __init__(self, n_cams=2, crop=200, n_kp=64, backbone='resnet18'):
        super().__init__()
        self.crop, self.backbone = crop, backbone
        if backbone == 'resnet18':
            self.nets = nn.ModuleList()
            for _ in range(n_cams):
                r = torchvision.models.resnet18(weights=None, norm_layer=lambda c: nn.GroupNorm(c // 16, c))
                r.avgpool, r.fc = SpatialSoftmax(512, n_kp), nn.Identity()
                self.nets.append(r)
            self.out_dim = n_cams * 2 * n_kp
        elif backbone == 'dinov2':
            import timm
            assert crop % 14 == 0, 'DINOv2 uses 14x14 patches'
            self.vit = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, img_size=crop)
            self.vit.requires_grad_(False).eval()
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.kp = nn.ModuleList([SpatialSoftmax(768, n_kp) for _ in range(n_cams)])
            self.cls = nn.ModuleList([nn.Linear(768, 2 * n_kp) for _ in range(n_cams)])
            self.out_dim = n_cams * 4 * n_kp
        else:
            raise ValueError(backbone)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone == 'dinov2':
            self.vit.eval()  # frozen: never leave eval, whatever the parent does
        return self

    def _camera(self, c, x):
        if self.backbone == 'resnet18':
            return self.nets[c](x)
        with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            tokens = self.vit.forward_features((x - self.mean) / self.std).float()
        cls, patches = tokens[:, 0], tokens[:, 1:]
        g = self.crop // 14
        fmap = patches.transpose(1, 2).reshape(len(x), -1, g, g)
        return torch.cat([self.kp[c](fmap), self.cls[c](cls)], dim=-1)

    def forward(self, images):
        b, n, h, w, _ = images.shape
        x = images.permute(0, 1, 4, 2, 3).float() / 255.0
        if self.training:
            i = torch.randint(0, h - self.crop + 1, (b,))
            j = torch.randint(0, w - self.crop + 1, (b,))
            x = torch.stack([x[k, :, :, i[k]:i[k] + self.crop, j[k]:j[k] + self.crop] for k in range(b)])
        else:
            i, j = (h - self.crop) // 2, (w - self.crop) // 2
            x = x[:, :, :, i:i + self.crop, j:j + self.crop]
        return torch.cat([self._camera(c, x[:, c]) for c in range(n)], dim=-1)
