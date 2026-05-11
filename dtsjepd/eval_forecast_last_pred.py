import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dtsjepd.datasets import load_series_splits
    from dtsjepd.models import MultivariateEncoder
except ImportError:
    from datasets import load_series_splits
    from models import load_series_splits, MultivariateEncoder


class ForecastWindowDataset(Dataset):
    """Build (context_patches -> next_patch) pairs from one split tensor.

    series: [T, C]
    context_patches: K, target: K+1-th patch
    """

    def __init__(self, series_tensor, patch_size=32, context_patches=10):
        self.patch_size = patch_size
        self.context_patches = context_patches

        num_full_patches = len(series_tensor) // patch_size
        trimmed = series_tensor[: num_full_patches * patch_size]
        self.patches = trimmed.view(num_full_patches, patch_size, -1)

    def __len__(self):
        return max(0, self.patches.size(0) - self.context_patches)

    def __getitem__(self, idx):
        context = self.patches[idx : idx + self.context_patches]
        target = self.patches[idx + self.context_patches]
        return context, target


class LinearDecoder(nn.Module):
    def __init__(self, emb_dim, patch_size, num_channels):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.fc = nn.Linear(emb_dim, patch_size * num_channels)

    def forward(self, x):
        out = self.fc(x)
        return out.view(x.size(0), self.patch_size, self.num_channels)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="weather")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--input_cols", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context_patches", type=int, default=10)
    parser.add_argument("--save_json_path", type=str, default="")
    parser.add_argument("--save_json_dir", type=str, default="")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_encoder_from_ckpt(ckpt, device):
    ckpt_args = ckpt.get("args", {})
    num_patches = ckpt.get("num_patches")
    patch_size = ckpt.get("patch_size")
    num_channels = ckpt.get("num_channels")

    if num_patches is None or patch_size is None or num_channels is None:
        raise ValueError("Checkpoint missing num_patches/patch_size/num_channels metadata.")

    encoder = MultivariateEncoder(
        num_patches=num_patches,
        patch_size=patch_size,
        num_channels=num_channels,
        kernel_size=ckpt_args["encoder_kernel_size"],
        embed_dim=ckpt_args["encoder_embed_dim"],
        nhead=ckpt_args["encoder_nhead"],
        num_layers=ckpt_args["encoder_num_layers"],
        jepa=True,
    )
    encoder.load_state_dict(ckpt["encoder"], strict=True)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    return encoder, num_patches, patch_size, num_channels, ckpt_args["encoder_embed_dim"]


def encode_context_sum(encoder, context_patches):
    with torch.no_grad():
        emb = encoder(context_patches)
    return torch.sum(emb, dim=1)


def evaluate_test(encoder, decoder, test_series, patch_size, context_patches, device):
    encoder.eval()
    decoder.eval()

    num_steps = max(0, (len(test_series) - context_patches * patch_size) // patch_size)
    if num_steps == 0:
        raise ValueError("Test split too short for requested context/patch settings.")

    mse_list = []
    mae_list = []

    with torch.no_grad():
        for step in range(num_steps):
            start = step * patch_size
            end = start + context_patches * patch_size
            target_start = end
            target_end = target_start + patch_size

            current_context = test_series[start:end].view(1, context_patches, patch_size, -1).to(device)
            target_patch = test_series[target_start:target_end].view(1, patch_size, -1).to(device)

            summed_embedding = encode_context_sum(encoder, current_context)
            pred_patch = decoder(summed_embedding)

            diff = pred_patch - target_patch
            mse_list.append(torch.mean(diff ** 2).item())
            mae_list.append(torch.mean(torch.abs(diff)).item())

    return float(np.mean(mse_list)), float(np.mean(mae_list)), num_steps


def compute_train_target_mean(train_dataset):
    targets = []
    for _, target_patch in train_dataset:
        targets.append(target_patch)
    if not targets:
        raise ValueError("Training dataset has no targets for mean baseline.")
    return torch.stack(targets, dim=0).mean(dim=0)


def evaluate_mean_baseline(mean_patch, test_series, patch_size, context_patches, device):
    num_steps = max(0, (len(test_series) - context_patches * patch_size) // patch_size)
    if num_steps == 0:
        raise ValueError("Test split too short for requested context/patch settings.")

    mean_patch = mean_patch.to(device).view(1, patch_size, -1)
    mse_list = []
    mae_list = []

    with torch.no_grad():
        for step in range(num_steps):
            target_start = (step + context_patches) * patch_size
            target_end = target_start + patch_size
            target_patch = test_series[target_start:target_end].view(1, patch_size, -1).to(device)

            diff = mean_patch - target_patch
            mse_list.append(torch.mean(diff ** 2).item())
            mae_list.append(torch.mean(torch.abs(diff)).item())

    return float(np.mean(mse_list)), float(np.mean(mae_list)), num_steps


def evaluate_naive_baseline(test_series, patch_size, context_patches, device):
    """Persistence baseline: predict the next patch with the last context patch."""
    num_steps = max(0, (len(test_series) - context_patches * patch_size) // patch_size)
    if num_steps == 0:
        raise ValueError("Test split too short for requested context/patch settings.")

    mse_list = []
    mae_list = []

    with torch.no_grad():
        for step in range(num_steps):
            context_start = step * patch_size
            last_context_start = context_start + (context_patches - 1) * patch_size
            last_context_end = last_context_start + patch_size
            target_start = context_start + context_patches * patch_size
            target_end = target_start + patch_size

            pred_patch = test_series[last_context_start:last_context_end].view(1, patch_size, -1).to(device)
            target_patch = test_series[target_start:target_end].view(1, patch_size, -1).to(device)

            diff = pred_patch - target_patch
            mse_list.append(torch.mean(diff ** 2).item())
            mae_list.append(torch.mean(torch.abs(diff)).item())

    return float(np.mean(mse_list)), float(np.mean(mae_list)), num_steps


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    encoder, num_patches, patch_size, num_channels, emb_dim = build_encoder_from_ckpt(ckpt, device)

    if args.context_patches != num_patches:
        print(
            f"Warning: context_patches={args.context_patches} but checkpoint num_patches={num_patches}. "
            f"Using context_patches={num_patches} to match encoder positional setup."
        )
    context_patches = num_patches

    data_path = args.data_path or f"./data/{args.data}/{args.data}.csv"
    splits = load_series_splits(data_path, args.data, input_cols=args.input_cols)

    train_dataset = ForecastWindowDataset(
        splits["train"], patch_size=patch_size, context_patches=context_patches
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    decoder = LinearDecoder(emb_dim=emb_dim, patch_size=patch_size, num_channels=num_channels).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    history = []
    for epoch in range(1, args.num_epochs + 1):
        encoder.eval()
        decoder.train()
        running_loss = 0.0
        n_batches = 0

        for context_patches_batch, target_patch in train_loader:
            context_patches_batch = context_patches_batch.to(device)
            target_patch = target_patch.to(device)

            summed_embedding = encode_context_sum(encoder, context_patches_batch)
            pred_patch = decoder(summed_embedding)
            loss = criterion(pred_patch, target_patch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            raise ValueError("Training loader has zero batches.")
        train_mse = running_loss / n_batches
        history.append({"epoch": epoch, "train_mse": train_mse})

        if epoch % 10 == 0 or epoch == 1 or epoch == args.num_epochs:
            print(f"[Epoch {epoch:03d}] train_mse={train_mse:.6f}")

    test_mse, test_mae, test_steps = evaluate_test(
        encoder,
        decoder,
        splits["test"],
        patch_size=patch_size,
        context_patches=context_patches,
        device=device,
    )
    mean_patch = compute_train_target_mean(train_dataset)
    mean_mse, mean_mae, mean_steps = evaluate_mean_baseline(
        mean_patch,
        splits["test"],
        patch_size=patch_size,
        context_patches=context_patches,
        device=device,
    )
    naive_mse, naive_mae, naive_steps = evaluate_naive_baseline(
        splits["test"],
        patch_size=patch_size,
        context_patches=context_patches,
        device=device,
    )
    print(f"Test MSE is: {test_mse}")
    print(f"Test MAE is: {test_mae}")
    print(f"Mean baseline MSE is: {mean_mse}")
    print(f"Mean baseline MAE is: {mean_mae}")
    print(f"Naive baseline MSE is: {naive_mse}")
    print(f"Naive baseline MAE is: {naive_mae}")

    result = {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "data_path": data_path,
        "selected_cols": splits["selected_cols"],
        "seed": args.seed,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patch_size": patch_size,
        "context_patches": context_patches,
        "num_channels": num_channels,
        "train_windows": len(train_dataset),
        "test_steps": test_steps,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "mean_baseline_mse": mean_mse,
        "mean_baseline_mae": mean_mae,
        "naive_baseline_mse": naive_mse,
        "naive_baseline_mae": naive_mae,
        "metrics": {
            "test": {
                "mse": test_mse,
                "mae": test_mae,
                "steps": test_steps,
            },
            "mean_baseline": {
                "mse": mean_mse,
                "mae": mean_mae,
                "steps": mean_steps,
            },
            "naive_baseline": {
                "mse": naive_mse,
                "mae": naive_mae,
                "steps": naive_steps,
            }
        },
        "history": history,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    save_json_path = args.save_json_path.strip()
    save_json_dir = args.save_json_dir.strip()
    if save_json_path:
        out_path = Path(save_json_path)
    elif save_json_dir:
        ckpt_stem = Path(args.checkpoint).stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(save_json_dir) / f"{ckpt_stem}_forecast_last_pred_{ts}.json"
    else:
        out_path = None

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved eval result JSON to: {out_path}")


if __name__ == "__main__":
    main()
