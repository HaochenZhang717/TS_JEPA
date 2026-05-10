import math

import numpy as np
import torch
import torch.nn as nn

from src.models.utils.mask_utils import apply_mask
from src.models.utils.modules import Block


class MultivariateTSTokenizer(nn.Module):
    def __init__(self, num_channels, patch_size, kernel_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels=num_channels,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            padding=0,
        )
        conv_output_length = (patch_size - kernel_size) // kernel_size + 1
        self.fc = nn.Linear(embed_dim * conv_output_length, embed_dim)

    def forward(self, x):
        batch_size, num_patches, patch_size, num_channels = x.shape
        x = x.reshape(batch_size * num_patches, patch_size, num_channels)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x.reshape(batch_size, num_patches, -1)


class MultivariateEncoder(nn.Module):
    def __init__(
        self,
        num_patches,
        patch_size,
        num_channels,
        kernel_size,
        embed_dim,
        nhead,
        num_layers,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        jepa=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.jepa = jepa

        self.tokenizer = MultivariateTSTokenizer(
            num_channels=num_channels,
            patch_size=patch_size,
            kernel_size=kernel_size,
            embed_dim=embed_dim,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )
        self.init_embed()

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=nhead,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    act_layer=nn.GELU,
                    norm_layer=norm_layer,
                )
                for _ in range(num_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        batch_size, num_patches, _, _ = x.shape
        x = self.tokenizer(x)
        pos_embs = self.pos_embed.repeat(batch_size, 1, 1)[:, :num_patches, :]
        x = x + pos_embs

        if mask is not None and self.jepa:
            x = apply_mask(x, mask)

        for block in self.blocks:
            x = block(x, mask=None)

        return self.encoder_norm(x)

    def init_embed(self):
        assert self.embed_dim % 2 == 0

        omega = np.arange(self.embed_dim // 2, dtype=float)
        omega /= self.embed_dim / 2.0
        omega = 1.0 / 10000**omega

        pos = np.arange(self.num_patches, dtype=float)
        out = np.einsum("m,d->md", pos, omega)
        emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
        self.pos_embed.data.copy_(torch.from_numpy(emb).float().unsqueeze(0))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=128):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class AdaLNPatchDenoisingHead(nn.Module):
    def __init__(
        self,
        patch_size,
        num_channels,
        predictor_embed_dim,
        hidden_dim=128,
        time_frequency_dim=128,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv1d(
            in_channels=num_channels,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.t_embedder = TimestepEmbedder(hidden_dim, time_frequency_dim)
        self.predictor_embedder = nn.Linear(predictor_embed_dim, hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.activation = nn.GELU()
        self.out = nn.Linear(patch_size * hidden_dim, patch_size * num_channels)

        self.reset_adaln_parameters()

    def reset_adaln_parameters(self):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, z_t, t, condition):
        batch_size, num_masked, patch_size, num_channels = z_t.shape
        z_t = z_t.reshape(batch_size * num_masked, patch_size, num_channels)
        z_t = z_t.permute(0, 2, 1)

        condition = condition.reshape(batch_size * num_masked, -1)
        t = t.reshape(batch_size * num_masked)

        cond = self.t_embedder(t) + self.predictor_embedder(condition)

        features = self.conv(z_t)
        features = features.transpose(1, 2)
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        features = self.norm(features)
        features = features * (1 + scale[:, None, :]) + shift[:, None, :]
        features = self.activation(features)

        out = self.out(features.reshape(features.size(0), -1))
        return out.reshape(batch_size, num_masked, patch_size, num_channels)


def init_dtsjepd_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
