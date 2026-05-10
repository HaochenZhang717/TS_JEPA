"""
    Multivariate TS-JEPA pretraining.

    This script keeps the original JEPA objective but lets each patch contain
    multiple channels: [batch, num_patches, patch_size, num_channels].
"""

import argparse
import copy
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from config.config_pretrain import config as base_config
from main.utils import init_weights
from src.models.predictor import Predictor
from src.models.utils.mask_utils import apply_mask
from src.models.utils.modules import Block

warnings.filterwarnings("ignore")


class MultivariateCSVDataLoader:
    def __init__(
        self,
        path_data,
        data_name,
        input_cols="auto",
        series_split_size=320,
        patch_size=32,
        mask_ratio=0.7,
    ):
        timestamp_col = "date"
        validation_fraction = 0.05
        test_fraction = 0.3

        df = pd.read_csv(path_data, low_memory=False)
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df.sort_values(by=[timestamp_col], inplace=True)

        selected_cols = self._resolve_input_cols(df, data_name, input_cols)
        values = df[selected_cols].apply(pd.to_numeric, errors="coerce")
        if values.isna().any().any():
            missing = values.isna().sum()
            missing = missing[missing > 0].to_dict()
            raise ValueError(f"Selected columns contain NaNs: {missing}")

        val_len = int(len(values) * validation_fraction)
        test_len = int(len(values) * test_fraction)
        train_len = len(values) - val_len - test_len

        train_values = values.iloc[:train_len]
        val_values = values.iloc[train_len : train_len + val_len]
        test_values = values.iloc[train_len + val_len :]

        # Fit normalization on train only to avoid leaking validation/test stats.
        train_mean = train_values.mean(0)
        train_std = train_values.std(0).replace(0, 1.0)

        self.train_df = torch.tensor(
            ((train_values - train_mean) / train_std).values
        ).float()
        self.val_df = torch.tensor(((val_values - train_mean) / train_std).values).float()
        self.test_df = torch.tensor(
            ((test_values - train_mean) / train_std).values
        ).float()

        self.selected_cols = selected_cols
        self.series_split_size = series_split_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.time_series_list = self.train_df

    @staticmethod
    def _resolve_input_cols(df, data_name, input_cols):
        numeric_cols = [
            col
            for col in df.columns
            if col != "date" and pd.api.types.is_numeric_dtype(df[col])
        ]

        if input_cols == "auto":
            household_cols = [str(i) for i in range(6)]
            if data_name.lower() == "household" and all(
                col in numeric_cols for col in household_cols
            ):
                return household_cols
            return numeric_cols

        if input_cols == "all":
            return numeric_cols

        selected_cols = [col.strip() for col in input_cols.split(",") if col.strip()]
        missing = [col for col in selected_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Requested input columns are missing: {missing}")
        return selected_cols

    def __getitem__(self, idx):
        ts = self.time_series_list
        num_splits = len(ts) // self.series_split_size
        split_series = [
            ts[i * self.series_split_size : (i + 1) * self.series_split_size]
            for i in range(num_splits)
        ]
        selected_series = split_series[idx % len(split_series)]

        num_patches = len(selected_series) // self.patch_size
        patches = [
            selected_series[i * self.patch_size : (i + 1) * self.patch_size]
            for i in range(num_patches)
        ]
        patches_tensor = torch.stack(patches)

        num_masked_patches = int(num_patches * self.mask_ratio)
        mask_indices = random.sample(range(num_patches), num_masked_patches)
        non_mask_indices = [i for i in range(num_patches) if i not in mask_indices]

        return (
            patches_tensor,
            torch.tensor(mask_indices),
            torch.tensor(non_mask_indices),
        )

    def __len__(self):
        return len(self.time_series_list) // self.series_split_size


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="weather")
    parser.add_argument("--input_cols", type=str, default="auto")
    parser.add_argument("--mask_ratio", type=float, default=base_config["mask_ratio"])
    parser.add_argument("--batch_size", type=int, default=base_config["batch_size"])
    parser.add_argument("--lr", type=float, default=base_config["lr"])
    parser.add_argument("--ema_momentum", type=float, default=base_config["ema_momentum"])
    parser.add_argument("--ratio_patches", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=base_config["num_epochs"])
    parser.add_argument("--checkpoint_save", type=int, default=base_config["checkpoint_save"])
    parser.add_argument("--save_suffix", type=str, default="")
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project_name", type=str, default="")
    parser.add_argument("--encoder_embed_dim", type=int, default=128)
    parser.add_argument("--encoder_nhead", type=int, default=2)
    parser.add_argument("--encoder_num_layers", type=int, default=1)
    parser.add_argument("--encoder_kernel_size", type=int, default=3)
    parser.add_argument("--predictor_embed", type=int, default=128)
    parser.add_argument("--predictor_nhead", type=int, default=2)
    parser.add_argument("--predictor_num_layers", type=int, default=1)
    return parser.parse_args()


def loss_pred(pred, target_ema):
    return torch.mean(torch.abs(pred - target_ema))


def build_path_save(args, num_channels):
    path_save = (
        f"./logs/output_model/{args.data}/multivariate"
        f"_lr_{args.lr}"
        f"_ema_momentum_{args.ema_momentum}"
        f"_mask_ratio_{args.mask_ratio}"
        f"_ratio_patches_{args.ratio_patches}"
        f"_channels_{num_channels}"
        f"_encoder_{args.encoder_embed_dim}_{args.encoder_nhead}_{args.encoder_num_layers}"
        f"_predictor_{args.predictor_embed}_{args.predictor_nhead}_{args.predictor_num_layers}"
    )
    if args.save_suffix:
        path_save = path_save + "_" + args.save_suffix
    return path_save


def save_model(path_save, model, epoch, args, selected_cols):
    save_dict = {
        "encoder": model.state_dict(),
        "epoch": epoch,
        "selected_cols": selected_cols,
        "args": vars(args),
    }
    path_name = path_save + "_epoch_" + str(epoch) + ".pt"
    os.makedirs(os.path.dirname(path_name), exist_ok=True)
    torch.save(save_dict, path_name)


if __name__ == "__main__":
    args = parse_args()

    seed = random.randint(0, 100)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path_data = f"./data/{args.data}/{args.data}.csv"

    dataset = MultivariateCSVDataLoader(
        path_data=path_data,
        data_name=args.data,
        input_cols=args.input_cols,
        series_split_size=args.ratio_patches * 32,
        patch_size=32,
        mask_ratio=args.mask_ratio,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    sample_patches, _, _ = dataset[0]
    num_patches, patch_size, num_channels = sample_patches.shape
    path_save = build_path_save(args, num_channels)

    print(f"Using columns ({num_channels}): {dataset.selected_cols}")
    print(f"Checkpoint prefix: {path_save}")

    encoder = MultivariateEncoder(
        num_patches=num_patches,
        patch_size=patch_size,
        num_channels=num_channels,
        kernel_size=args.encoder_kernel_size,
        embed_dim=args.encoder_embed_dim,
        nhead=args.encoder_nhead,
        num_layers=args.encoder_num_layers,
        jepa=True,
    )
    predictor = Predictor(
        num_patches=num_patches,
        encoder_embed_dim=args.encoder_embed_dim,
        predictor_embed_dim=args.predictor_embed,
        nhead=args.predictor_nhead,
        num_layers=args.predictor_num_layers,
    )

    for module in encoder.modules():
        init_weights(module)
    for module in predictor.modules():
        init_weights(module)

    optimizer = torch.optim.AdamW(
        [
            {"params": (p for _, p in encoder.named_parameters())},
            {"params": (p for _, p in predictor.named_parameters())},
        ],
        lr=args.lr,
    )
    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.5, total_iters=args.num_epochs
    )

    encoder = encoder.to(device)
    predictor = predictor.to(device)
    encoder_ema = copy.deepcopy(encoder)
    for param in encoder_ema.parameters():
        param.requires_grad = False

    ema_scheduler = (
        args.ema_momentum
        + i * (1 - args.ema_momentum) / (args.num_epochs * base_config["ipe_scale"])
        for i in range(int(args.num_epochs * base_config["ipe_scale"]) + 1)
    )

    wandb_run = None
    if args.log_wandb:
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "Weights & Biases logging is enabled but wandb is not installed. "
                "Install it with `pip install wandb`."
            )
        wandb_run = wandb.init(
            project=args.wandb_project_name or f"{args.data}_multivariate_pretrain",
            config={**vars(args), "seed": seed, "selected_cols": dataset.selected_cols},
            name=(
                f"{args.data}_mv{num_channels}_lr{args.lr}_bs{args.batch_size}"
                f"_m{args.ema_momentum}_mask{args.mask_ratio}"
                f"_enc{args.encoder_embed_dim}-{args.encoder_nhead}-{args.encoder_num_layers}"
                f"_pred{args.predictor_embed}-{args.predictor_nhead}-{args.predictor_num_layers}"
            ),
        )

    num_batches = len(loader)
    save_model(path_save, encoder, 0, args, dataset.selected_cols)

    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        m = next(ema_scheduler)
        encoder.train()
        predictor.train()
        total_loss = 0.0

        for patches, masks, non_masks in loader:
            optimizer.zero_grad()
            patches = patches.to(device)
            masks = masks.to(device)
            non_masks = non_masks.to(device)

            with torch.no_grad():
                target_ema = encoder_ema(patches)
                target_ema = F.layer_norm(target_ema, (target_ema.size(-1),))
                target_ema = apply_mask(target_ema, masks)

            tokens = encoder(patches, mask=non_masks)
            pred = predictor(tokens, mask=masks, non_masks=non_masks)
            loss = loss_pred(pred, target_ema)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for param_q, param_k in zip(
                    encoder.parameters(), encoder_ema.parameters()
                ):
                    param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

            total_loss += loss

        scheduler.step()
        total_loss = total_loss / num_batches

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}, lr: {optimizer.param_groups[0]['lr']:.3g} "
                f"- Multivariate JEPA Loss: {total_loss:.4f},"
            )

        if wandb_run is not None:
            import wandb

            wandb.log(
                {
                    "epoch": epoch,
                    "train/jepa_loss": total_loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/ema_momentum": m,
                    "train/epoch_time_sec": time.time() - epoch_start,
                    "data/num_channels": num_channels,
                }
            )

        if epoch % args.checkpoint_save == 0 and epoch != 0:
            save_model(path_save, encoder, epoch, args, dataset.selected_cols)

    if wandb_run is not None:
        wandb.finish()
