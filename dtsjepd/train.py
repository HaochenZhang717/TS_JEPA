import argparse
import copy
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dtsjepd.datasets import build_dataloaders, gather_masked_patches
    from dtsjepd.models import (
        AdaLNPatchDenoisingHead,
        MultivariateEncoder,
        init_dtsjepd_weights,
    )
except ImportError:
    from datasets import build_dataloaders, gather_masked_patches
    from models import AdaLNPatchDenoisingHead, MultivariateEncoder, init_dtsjepd_weights

from src.models.predictor import Predictor
from src.models.utils.mask_utils import apply_mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="household")
    parser.add_argument("--input_cols", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=5001)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--ema_momentum", type=float, default=0.998)
    parser.add_argument("--mask_ratio", type=float, default=0.7)
    parser.add_argument("--ratio_patches", type=int, default=10)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--checkpoint_save", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--disable_eval", action="store_true")
    parser.add_argument("--clip_grad", type=float, default=10.0)
    parser.add_argument("--save_suffix", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project_name", type=str, default="TS_JEPA")

    parser.add_argument("--encoder_embed_dim", type=int, default=128)
    parser.add_argument("--encoder_nhead", type=int, default=2)
    parser.add_argument("--encoder_num_layers", type=int, default=1)
    parser.add_argument("--encoder_kernel_size", type=int, default=3)
    parser.add_argument("--predictor_embed", type=int, default=128)
    parser.add_argument("--predictor_nhead", type=int, default=2)
    parser.add_argument("--predictor_num_layers", type=int, default=1)

    parser.add_argument("--lambda_denoise", type=float, default=0.01)
    parser.add_argument("--denoise_hidden_dim", type=int, default=128)
    parser.add_argument("--time_frequency_dim", type=int, default=128)
    parser.add_argument("--P_mean", type=float, default=0.0)
    parser.add_argument("--P_std", type=float, default=1.0)
    parser.add_argument("--t_eps", type=float, default=1e-5)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--detach_denoise_condition", action="store_true")
    return parser.parse_args()


def jepa_loss(pred, target_ema):
    return torch.mean(torch.abs(pred - target_ema))


def sample_t(shape, device, p_mean, p_std):
    z = torch.randn(shape, device=device) * p_std + p_mean
    return torch.sigmoid(z)


def denoise_loss(head, clean_patches, condition, args):
    if args.detach_denoise_condition:
        condition = condition.detach()

    t = sample_t(
        (clean_patches.size(0), clean_patches.size(1)),
        clean_patches.device,
        args.P_mean,
        args.P_std,
    )
    t_view = t.view(t.size(0), t.size(1), 1, 1)
    noise = torch.randn_like(clean_patches) * args.noise_scale
    z_t = t_view * clean_patches + (1 - t_view) * noise

    denom = (1 - t_view).clamp_min(args.t_eps)
    v_target = (clean_patches - z_t) / denom
    x_pred = head(z_t, t, condition)
    v_pred = (x_pred - z_t) / denom
    return torch.mean((v_target - v_pred) ** 2)


def make_path_save(args, num_channels):
    path = (
        f"./logs/output_model_dtsjepd/{args.data}/dtsjepd"
        f"_lr_{args.lr}"
        f"_ema_momentum_{args.ema_momentum}"
        f"_mask_ratio_{args.mask_ratio}"
        f"_ratio_patches_{args.ratio_patches}"
        f"_channels_{num_channels}"
        f"_lambda_{args.lambda_denoise}"
        f"_denoise_{args.denoise_hidden_dim}"
        f"_encoder_{args.encoder_embed_dim}_{args.encoder_nhead}_{args.encoder_num_layers}"
        f"_predictor_{args.predictor_embed}_{args.predictor_nhead}_{args.predictor_num_layers}"
    )
    if args.save_suffix:
        path = path + "_" + args.save_suffix
    return path


def build_warmup_constant_scheduler(optimizer, warmup_epochs):
    if warmup_epochs <= 0:
        return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    def lr_lambda(epoch):
        return min(1.0, float(epoch + 1) / float(warmup_epochs))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(path_save, epoch, encoder, predictor, denoising_head, args, metadata):
    path_name = path_save + "_epoch_" + str(epoch) + ".pt"
    os.makedirs(os.path.dirname(path_name), exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "denoising_head": denoising_head.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            **metadata,
        },
        path_name,
    )


def build_models(args, num_patches, patch_size, num_channels, device):
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
    denoising_head = AdaLNPatchDenoisingHead(
        patch_size=patch_size,
        num_channels=num_channels,
        predictor_embed_dim=args.encoder_embed_dim,
        hidden_dim=args.denoise_hidden_dim,
        time_frequency_dim=args.time_frequency_dim,
    )

    for module in encoder.modules():
        init_dtsjepd_weights(module)
    for module in predictor.modules():
        init_dtsjepd_weights(module)
    for module in denoising_head.modules():
        init_dtsjepd_weights(module)
    denoising_head.reset_adaln_parameters()

    return encoder.to(device), predictor.to(device), denoising_head.to(device)


def run_epoch(
    loader,
    encoder,
    predictor,
    encoder_ema,
    denoising_head,
    args,
    device,
    optimizer=None,
):
    is_train = optimizer is not None
    encoder.train(is_train)
    predictor.train(is_train)
    denoising_head.train(is_train)

    totals = {
        "jepa_loss": 0.0,
        "denoise_loss": 0.0,
        "total_loss": 0.0,
        "pred_embedding_norm": 0.0,
        "target_embedding_norm": 0.0,
        "denoise_condition_norm": 0.0,
        "grad_norm": 0.0,
    }
    num_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for patches, masks, non_masks in loader:
            patches = patches.to(device)
            masks = masks.to(device)
            non_masks = non_masks.to(device)

            if is_train:
                optimizer.zero_grad()

            with torch.no_grad():
                target_ema = encoder_ema(patches)
                target_ema = F.layer_norm(target_ema, (target_ema.size(-1),))
                target_ema = apply_mask(target_ema, masks)

            tokens = encoder(patches, mask=non_masks)
            pred = predictor(tokens, mask=masks, non_masks=non_masks)

            clean_masked = gather_masked_patches(patches, masks)
            loss_jepa = jepa_loss(pred, target_ema)

            loss_denoise = denoise_loss(denoising_head, clean_masked, pred, args)
            loss_total = loss_jepa + args.lambda_denoise * loss_denoise

            if is_train:
                loss_total.backward()
                grad_norm = 0.0
                if args.clip_grad and args.clip_grad > 0:
                    trainable_params = [
                        param
                        for group in optimizer.param_groups
                        for param in group["params"]
                        if param.grad is not None
                    ]
                    if trainable_params:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            trainable_params, args.clip_grad
                        ).item()
                optimizer.step()

                with torch.no_grad():
                    for param_q, param_k in zip(
                        encoder.parameters(), encoder_ema.parameters()
                    ):
                        param_k.data.mul_(args.ema_momentum).add_(
                            (1.0 - args.ema_momentum) * param_q.detach().data
                        )

            totals["jepa_loss"] += loss_jepa.detach().item()
            totals["denoise_loss"] += loss_denoise.detach().item()
            totals["total_loss"] += loss_total.detach().item()
            totals["pred_embedding_norm"] += pred.detach().norm(dim=-1).mean().item()
            totals["target_embedding_norm"] += (
                target_ema.detach().norm(dim=-1).mean().item()
            )
            totals["denoise_condition_norm"] += pred.detach().norm(dim=-1).mean().item()
            if is_train:
                totals["grad_norm"] += grad_norm
            num_batches += 1

    if num_batches == 0:
        raise ValueError("Dataloader produced zero batches.")

    return {key: value / num_batches for key, value in totals.items()}


def make_wandb_name(args, num_channels):
    return (
        f"{args.data}_dtsjepd_ch{num_channels}"
        f"_lr{args.lr}_bs{args.batch_size}"
        f"_m{args.ema_momentum}_mask{args.mask_ratio}"
        f"_lam{args.lambda_denoise}_dh{args.denoise_hidden_dim}"
        f"_enc{args.encoder_embed_dim}-{args.encoder_nhead}-{args.encoder_num_layers}"
        f"_pred{args.predictor_embed}-{args.predictor_nhead}-{args.predictor_num_layers}"
    )


def main():
    args = parse_args()
    seed = random.randint(0, 100)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path_data = f"./data/{args.data}/{args.data}.csv"

    train_loader, val_loader, _, data_meta = build_dataloaders(
        path_data=path_data,
        data_name=args.data,
        batch_size=args.batch_size,
        input_cols=args.input_cols,
        ratio_patches=args.ratio_patches,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        stride=args.stride if args.stride > 0 else None,
        num_workers=args.num_workers,
    )

    sample_patches, _, _ = train_loader.dataset[0]
    num_patches, patch_size, num_channels = sample_patches.shape
    encoder, predictor, denoising_head = build_models(
        args, num_patches, patch_size, num_channels, device
    )
    encoder_ema = copy.deepcopy(encoder)
    for param in encoder_ema.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        [
            {"params": (p for _, p in encoder.named_parameters())},
            {"params": (p for _, p in predictor.named_parameters())},
            {"params": (p for _, p in denoising_head.named_parameters())},
        ],
        lr=args.lr,
    )
    scheduler = build_warmup_constant_scheduler(optimizer, args.warmup_epochs)

    metadata = {
        "selected_cols": data_meta["selected_cols"],
        "train_mean": data_meta["train_mean"],
        "train_std": data_meta["train_std"],
        "num_channels": num_channels,
        "patch_size": patch_size,
        "num_patches": num_patches,
        "stride": args.stride if args.stride > 0 else args.ratio_patches * args.patch_size,
        "train_dataset_len": len(train_loader.dataset),
        "val_dataset_len": len(val_loader.dataset),
        "seed": seed,
    }
    path_save = make_path_save(args, num_channels)
    print(f"Using columns ({num_channels}): {data_meta['selected_cols']}")
    print(f"Checkpoint prefix: {path_save}")

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
            project=args.wandb_project_name,
            config={**vars(args), **metadata},
            name=make_wandb_name(args, num_channels),
        )

    save_checkpoint(path_save, 0, encoder, predictor, denoising_head, args, metadata)

    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        train_stats = run_epoch(
            train_loader,
            encoder,
            predictor,
            encoder_ema,
            denoising_head,
            args,
            device,
            optimizer=optimizer,
        )

        log_dict = {
            "epoch": epoch,
            "train/jepa_loss": train_stats["jepa_loss"],
            "train/denoise_loss": train_stats["denoise_loss"],
            "train/total_loss": train_stats["total_loss"],
            "train/weighted_denoise_loss": args.lambda_denoise
            * train_stats["denoise_loss"],
            "train/denoise_to_jepa_ratio": (
                args.lambda_denoise
                * train_stats["denoise_loss"]
                / max(train_stats["jepa_loss"], 1e-12)
            ),
            "train/lr": current_lr,
            "train/ema_momentum": args.ema_momentum,
            "train/lambda_denoise": args.lambda_denoise,
            "train/epoch_time_sec": time.time() - epoch_start,
            "train/pred_embedding_norm": train_stats["pred_embedding_norm"],
            "train/target_embedding_norm": train_stats["target_embedding_norm"],
            "train/denoise_condition_norm": train_stats["denoise_condition_norm"],
            "train/grad_norm": train_stats["grad_norm"],
            "data/num_channels": num_channels,
            "data/train_dataset_len": len(train_loader.dataset),
            "data/val_dataset_len": len(val_loader.dataset),
        }

        should_eval = (
            not args.disable_eval
            and args.eval_every > 0
            and (epoch % args.eval_every == 0 or epoch == args.num_epochs - 1)
        )
        if should_eval:
            val_stats = run_epoch(
                val_loader,
                encoder,
                predictor,
                encoder_ema,
                denoising_head,
                args,
                device,
                optimizer=None,
            )
            log_dict.update(
                {
                    "val/jepa_loss": val_stats["jepa_loss"],
                    "val/denoise_loss": val_stats["denoise_loss"],
                    "val/total_loss": val_stats["total_loss"],
                    "val/weighted_denoise_loss": args.lambda_denoise
                    * val_stats["denoise_loss"],
                    "val/denoise_to_jepa_ratio": (
                        args.lambda_denoise
                        * val_stats["denoise_loss"]
                        / max(val_stats["jepa_loss"], 1e-12)
                    ),
                }
            )

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}, lr: {current_lr:.3g} "
                f"- total: {train_stats['total_loss']:.4f}, "
                f"jepa: {train_stats['jepa_loss']:.4f}, "
                f"denoise: {train_stats['denoise_loss']:.4f}"
            )

        if wandb_run is not None:
            import wandb

            wandb.log(log_dict)

        if epoch % args.checkpoint_save == 0 and epoch != 0:
            save_checkpoint(
                path_save, epoch, encoder, predictor, denoising_head, args, metadata
            )

        scheduler.step()

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
