import argparse
import json
import random
import sys
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
    from models import MultivariateEncoder

from src.models.predictor import Predictor


class Patch10PredictionDataset(Dataset):
    def __init__(self, series_tensor, ratio_patches=10, patch_size=32, stride=None):
        self.series = series_tensor
        self.ratio_patches = ratio_patches
        self.patch_size = patch_size
        self.series_split_size = ratio_patches * patch_size
        self.stride = stride or self.series_split_size
        if self.stride <= 0:
            raise ValueError("stride must be positive")

    def __len__(self):
        if len(self.series) < self.series_split_size:
            return 0
        return (len(self.series) - self.series_split_size) // self.stride + 1

    def __getitem__(self, idx):
        num_items = len(self)
        if num_items == 0:
            raise IndexError("Dataset contains no valid windows")
        start = (idx % num_items) * self.stride
        window = self.series[start : start + self.series_split_size]
        patches = window.reshape(self.ratio_patches, self.patch_size, -1)
        x_context = patches[:9]
        y_target = patches[9]
        return x_context, y_target


class MLPDecoder(nn.Module):
    def __init__(self, embed_dim, patch_size, num_channels, hidden_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, patch_size * num_channels),
        )

    def forward(self, embedding):
        out = self.net(embedding)
        return out.view(embedding.size(0), self.patch_size, self.num_channels)


class CNNMLPDecoder(nn.Module):
    def __init__(self, embed_dim, patch_size, num_channels, hidden_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.expand = nn.Linear(embed_dim, hidden_dim * patch_size)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.head = nn.Linear(hidden_dim * patch_size, patch_size * num_channels)

    def forward(self, embedding):
        x = self.expand(embedding)
        x = x.view(embedding.size(0), self.hidden_dim, self.patch_size)
        x = self.act(self.conv(x))
        x = x.reshape(x.size(0), -1)
        x = self.head(x)
        return x.view(embedding.size(0), self.patch_size, self.num_channels)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="weather")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--input_cols", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--decoder", type=str, default="mlp", choices=["mlp", "cnn_mlp"])
    parser.add_argument("--decoder_hidden_dim", type=int, default=256)
    parser.add_argument("--metric_on_original_scale", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_backbone_from_ckpt(ckpt, device):
    ckpt_args = ckpt.get("args", {})
    num_patches = ckpt.get("num_patches")
    patch_size = ckpt.get("patch_size")
    num_channels = ckpt.get("num_channels")

    if num_patches is None or patch_size is None or num_channels is None:
        raise ValueError("Checkpoint is missing num_patches/patch_size/num_channels metadata.")

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
    predictor = Predictor(
        num_patches=num_patches,
        encoder_embed_dim=ckpt_args["encoder_embed_dim"],
        predictor_embed_dim=ckpt_args["predictor_embed"],
        nhead=ckpt_args["predictor_nhead"],
        num_layers=ckpt_args["predictor_num_layers"],
    )
    encoder.load_state_dict(ckpt["encoder"], strict=True)
    predictor.load_state_dict(ckpt["predictor"], strict=True)

    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    for p in predictor.parameters():
        p.requires_grad = False
    return encoder, predictor, num_patches, patch_size, num_channels, ckpt_args["encoder_embed_dim"]


def make_loader(series_tensor, batch_size, ratio_patches, patch_size, stride, num_workers, shuffle):
    dataset = Patch10PredictionDataset(
        series_tensor=series_tensor,
        ratio_patches=ratio_patches,
        patch_size=patch_size,
        stride=stride,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def fetch_embedding(encoder, predictor, x_context, device):
    batch_size = x_context.size(0)
    patch_size = x_context.size(2)
    num_channels = x_context.size(3)

    full = torch.zeros(batch_size, 10, patch_size, num_channels, device=device)
    full[:, :9] = x_context.to(device)

    non_masks = torch.arange(0, 9, device=device).unsqueeze(0).repeat(batch_size, 1)
    masks = torch.full((batch_size, 1), 9, dtype=torch.long, device=device)

    with torch.no_grad():
        tokens = encoder(full, mask=non_masks)
        pred = predictor(tokens, mask=masks, non_masks=non_masks)
    return pred[:, 0, :]


def compute_metrics(y_pred, y_true):
    mae = torch.mean(torch.abs(y_pred - y_true))
    mse = torch.mean((y_pred - y_true) ** 2)
    return mae.item(), mse.item()


def evaluate(loader, encoder, predictor, decoder, device, mean_t, std_t, metric_on_original_scale):
    decoder.eval()
    total_mae = 0.0
    total_mse = 0.0
    total_batches = 0

    with torch.no_grad():
        for x_context, y_target in loader:
            x_context = x_context.to(device)
            y_target = y_target.to(device)
            emb = fetch_embedding(encoder, predictor, x_context, device)
            y_pred = decoder(emb)

            if metric_on_original_scale:
                y_pred = y_pred * std_t + mean_t
                y_target = y_target * std_t + mean_t

            mae, mse = compute_metrics(y_pred, y_target)
            total_mae += mae
            total_mse += mse
            total_batches += 1

    if total_batches == 0:
        raise ValueError("Evaluation loader has zero batches.")
    return total_mae / total_batches, total_mse / total_batches


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    encoder, predictor, num_patches, patch_size, num_channels, encoder_embed_dim = (
        build_backbone_from_ckpt(ckpt, device)
    )
    if num_patches != 10:
        raise ValueError(f"This eval expects num_patches=10, but checkpoint has {num_patches}.")

    data_path = args.data_path or f"./data/{args.data}/{args.data}.csv"
    splits = load_series_splits(data_path, args.data, input_cols=args.input_cols)

    col_order = splits["selected_cols"]
    mean = torch.tensor([splits["train_mean"][c] for c in col_order], dtype=torch.float32, device=device)
    std = torch.tensor([splits["train_std"][c] for c in col_order], dtype=torch.float32, device=device)
    mean = mean.view(1, 1, -1)
    std = std.view(1, 1, -1)

    stride = args.stride if args.stride > 0 else patch_size * num_patches
    train_loader = make_loader(
        splits["train"], args.batch_size, num_patches, patch_size, stride, args.num_workers, True
    )
    val_loader = make_loader(
        splits["val"], args.batch_size, num_patches, patch_size, stride, args.num_workers, False
    )
    test_loader = make_loader(
        splits["test"], args.batch_size, num_patches, patch_size, stride, args.num_workers, False
    )

    if args.decoder == "mlp":
        decoder = MLPDecoder(
            embed_dim=encoder_embed_dim,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_dim=args.decoder_hidden_dim,
        )
    elif args.decoder == "cnn_mlp":
        decoder = CNNMLPDecoder(
            embed_dim=encoder_embed_dim,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_dim=args.decoder_hidden_dim,
        )
    else:
        raise ValueError(f"Unknown decoder type: {args.decoder}")
    decoder = decoder.to(device)

    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None

    for epoch in range(1, args.num_epochs + 1):
        decoder.train()
        epoch_loss = 0.0
        n_batches = 0

        for x_context, y_target in train_loader:
            x_context = x_context.to(device)
            y_target = y_target.to(device)

            emb = fetch_embedding(encoder, predictor, x_context, device)
            y_pred = decoder(emb)
            loss = criterion(y_pred, y_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            raise ValueError("Training loader has zero batches.")
        train_loss = epoch_loss / n_batches

        val_mae, val_mse = evaluate(
            val_loader,
            encoder,
            predictor,
            decoder,
            device,
            mean,
            std,
            args.metric_on_original_scale,
        )
        print(
            f"[Epoch {epoch:03d}] train_mse={train_loss:.6f} val_mae={val_mae:.6f} val_mse={val_mse:.6f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best model state captured.")
    decoder.load_state_dict(best_state)

    val_mae, val_mse = evaluate(
        val_loader,
        encoder,
        predictor,
        decoder,
        device,
        mean,
        std,
        args.metric_on_original_scale,
    )
    test_mae, test_mse = evaluate(
        test_loader,
        encoder,
        predictor,
        decoder,
        device,
        mean,
        std,
        args.metric_on_original_scale,
    )

    result = {
        "checkpoint": args.checkpoint,
        "decoder": args.decoder,
        "decoder_hidden_dim": args.decoder_hidden_dim,
        "metric_on_original_scale": args.metric_on_original_scale,
        "best_val_mae": val_mae,
        "best_val_mse": val_mse,
        "test_mae": test_mae,
        "test_mse": test_mse,
        "data": args.data,
        "data_path": data_path,
        "num_channels": num_channels,
        "patch_size": patch_size,
        "num_patches": num_patches,
        "selected_cols": col_order,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
