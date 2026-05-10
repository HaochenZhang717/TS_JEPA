"""
Downstream forecasting evaluation using the pretrained encoder and predictor.

This variant uses the first nine context patches as input, asks the pretrained
predictor to infer the tenth patch embedding, then maps that embedding to the
forecast patch with a three-layer CNN head.
"""

from config.config_downstream import config

import os
import warnings

import numpy as np
import torch
import torch.nn as nn

from main.utils import prepare_args
from main.utils import mse, mae
from src.data_loaders.data_loader import get_evaluation_loaders
from src.models.encoder import Encoder
from src.models.predictor import Predictor

warnings.filterwarnings("ignore")


class CNNPatchDecoder(nn.Module):
    def __init__(self, emb_dim, patch_size=32, channels=None, kernel_size=3, dense_dim=32):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(channels[0], channels[1], kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(channels[1], channels[2], kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(dense_dim),
            nn.Flatten(),
            nn.Linear(channels[2] * dense_dim, patch_size),
        )

    def forward(self, predicted_embedding):
        return self.net(predicted_embedding.unsqueeze(1))


def _cnn_channels(config):
    channels = config.get("cnn_out_channels", [32, 64, 128])
    if isinstance(channels, int):
        return [channels, channels * 2, channels * 4]
    return channels


def _predict_next_embedding(encoder, predictor, context_patches):
    batch_size = context_patches.size(0)
    device = context_patches.device

    encoder_input = context_patches[:, :9, :]
    encoded_context = encoder(encoder_input)

    non_masks = torch.arange(9, device=device).unsqueeze(0).repeat(batch_size, 1)
    masks = torch.full((batch_size, 1), 9, dtype=torch.long, device=device)
    predicted_embedding = predictor(encoded_context, mask=masks, non_masks=non_masks)

    return predicted_embedding.squeeze(1)


if __name__ == "__main__":
    config = prepare_args(config)

    num_epochs = 500
    context = 32
    num_patches = 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    print("Load data")
    config["path_data"] = "./data/" + config["data"] + "/" + config["data"] + ".csv"
    loader = get_evaluation_loaders(
        config["path_data"],
        config["batch_size"],
        config["ratio_patches"],
        config["mask_ratio"],
    )

    input_dim = len(loader.dataset[0][0][0])
    num_context_patches = len(loader.dataset[0][0])

    encoder = Encoder(
        num_patches=num_context_patches,
        dim_in=input_dim,
        kernel_size=config["pretrain_encoder_kernel_size"],
        embed_dim=config["pretrain_encoder_embed_dim"],
        embed_bias=config["pretrain_encoder_embed_bias"],
        nhead=config["pretrain_encoder_nhead"],
        num_layers=config["pretrain_encoder_num_layers"],
        jepa=True,
    )

    predictor = Predictor(
        num_patches=num_context_patches,
        encoder_embed_dim=config["pretrain_encoder_embed_dim"],
        predictor_embed_dim=config["pretrain_decoder_embed_dim"],
        nhead=config["pretrain_decoder_nhead"],
        num_layers=config["pretrain_decoder_num_layers"],
    )

    cnn_decoder = CNNPatchDecoder(
        emb_dim=config["pretrain_encoder_embed_dim"],
        patch_size=context,
        channels=_cnn_channels(config),
        kernel_size=config["cnn_kernel_size"],
        dense_dim=config["cnn_dense_dim"],
    )

    encoder = encoder.to(device)
    predictor = predictor.to(device)
    cnn_decoder = cnn_decoder.to(device)

    checkpoint_path = config.get("checkpoint_path", "")
    if checkpoint_path:
        ckpt_to_load = checkpoint_path
    else:
        path_name = (
            "/lr_"
            + str(config["lr_pretrain"])
            + "_ema_momentum_"
            + str(config["ema_pretrain"])
            + "_mask_ratio_"
            + str(config["mask_ratio"])
            + "_ratio_patches_"
            + str(config["ratio_patches"])
            + "_encoder_"
            + str(config["pretrain_encoder_embed_dim"])
            + "_"
            + str(config["pretrain_encoder_nhead"])
            + "_"
            + str(config["pretrain_encoder_num_layers"])
            + "_predictor_"
            + str(config["pretrain_decoder_embed_dim"])
            + "_"
            + str(config["pretrain_decoder_nhead"])
            + "_"
            + str(config["pretrain_decoder_num_layers"])
            + "_epoch_"
            + str(config["checkpoint_to_use"])
        )
        ckpt_to_load = config["path_save"] + path_name + ".pt"

    checkpoint = torch.load(ckpt_to_load, map_location=device)
    if "predictor" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain predictor weights. "
            "Please rerun pretrain.py after the checkpoint-saving update."
        )
    encoder.load_state_dict(checkpoint["encoder"])
    predictor.load_state_dict(checkpoint["predictor"])
    print("Model loaded from: {}".format(ckpt_to_load))

    optimizer = torch.optim.AdamW(cnn_decoder.parameters(), lr=config["lr"])

    print("start train")
    for epoch in range(num_epochs):
        encoder.eval()
        predictor.eval()
        cnn_decoder.train()
        total_loss = 0.0

        for context_patches, target_patch in loader:
            context_patches = context_patches.to(device)
            target_patch = target_patch.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                predicted_context_embedding = _predict_next_embedding(
                    encoder, predictor, context_patches
                )
            predicted_next_patch = cnn_decoder(predicted_context_embedding)
            loss = torch.nn.functional.mse_loss(
                predicted_next_patch, target_patch, reduction="mean"
            )

            loss.backward()
            optimizer.step()

            total_loss += loss / config["batch_size"]

        if epoch % 10 == 0:
            print("Epoch: {} - Total loss: {}".format(epoch, total_loss))

    num_steps = (len(loader.dataset.test_df[context * num_patches :])) // context

    predictions = []
    targets = []
    contexts = []
    l_val_mse = []
    l_val_mae = []

    encoder.eval()
    predictor.eval()
    cnn_decoder.eval()
    with torch.no_grad():
        for step in range(num_steps):
            current_context = (
                loader.dataset.test_df[
                    context * step : context * num_patches + context * step
                ]
                .reshape(num_patches, context)
                .unsqueeze(0)
                .to(device)
            )
            target_value = loader.dataset.test_df[
                context * num_patches
                + step * context : context * num_patches
                + (step + 1) * context
            ].to(device)

            predicted_context_embedding = _predict_next_embedding(
                encoder, predictor, current_context
            )
            predicted_next_patch = cnn_decoder(predicted_context_embedding)

            y_pred = predicted_next_patch.flatten().detach().cpu().numpy()
            y_true = target_value.detach().cpu().numpy()

            l_val_mse.append(mse(y_pred, y_true))
            l_val_mae.append(mae(y_pred, y_true))
            predictions.append(y_pred)
            targets.append(y_true)
            contexts.append(current_context[:, :9, :].flatten().detach().cpu().numpy())

    print("MSE Loss is: {}".format(np.mean(l_val_mse)))
    print("MAE Loss is: {}".format(np.mean(l_val_mae)))

    if config.get("plot_path", ""):
        plot_num_steps = min(config.get("plot_num_steps", 20), len(predictions))
        y_pred = np.concatenate(predictions[:plot_num_steps])
        y_true = np.concatenate(targets[:plot_num_steps])

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            os.makedirs(os.path.dirname(config["plot_path"]), exist_ok=True)
            plt.figure(figsize=(14, 5))
            plt.plot(y_true, label="Target", linewidth=2)
            plt.plot(y_pred, label="Prediction", linewidth=2, alpha=0.85)
            for boundary in range(context, len(y_true), context):
                plt.axvline(boundary, color="0.88", linewidth=0.8)
            plt.title(
                "Mine forecast | lr_pretrain={} ema={} checkpoint={}".format(
                    config["lr_pretrain"],
                    config["ema_pretrain"],
                    config["checkpoint_to_use"],
                )
            )
            plt.xlabel("Forecast horizon index")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()
            plt.savefig(config["plot_path"], dpi=160)
            plt.close()
            print("Saved prediction plot to: {}".format(config["plot_path"]))

            example_plot_path = config["plot_path"].replace(".png", "_examples.png")
            num_examples = min(config.get("plot_num_examples", 12), len(predictions))
            example_indices = np.linspace(
                0, len(predictions) - 1, num=num_examples, dtype=int
            )
            ncols = 3
            nrows = int(np.ceil(num_examples / ncols))
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(5.2 * ncols, 3.1 * nrows),
                sharex=False,
                sharey=True,
            )
            axes = np.atleast_1d(axes).flatten()
            for ax, idx in zip(axes, example_indices):
                context_tail = contexts[idx][-context:]
                context_x = np.arange(-len(context_tail), 0)
                future_x = np.arange(len(targets[idx]))
                ax.plot(context_x, context_tail, color="0.55", linewidth=1.2, label="Context")
                ax.plot(future_x, targets[idx], color="#1f77b4", linewidth=2, label="Target")
                ax.plot(
                    future_x,
                    predictions[idx],
                    color="#ff7f0e",
                    linewidth=2,
                    linestyle="--",
                    label="Prediction",
                )
                ax.axvline(0, color="0.2", linewidth=1)
                ax.set_title("Sample step {}".format(idx))
                ax.grid(alpha=0.2)
            for ax in axes[num_examples:]:
                ax.axis("off")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=3)
            fig.suptitle(
                "Mine forecast samples | lr_pretrain={} ema={} checkpoint={}".format(
                    config["lr_pretrain"],
                    config["ema_pretrain"],
                    config["checkpoint_to_use"],
                ),
                y=0.995,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.savefig(example_plot_path, dpi=160)
            plt.close(fig)
            print("Saved forecast examples plot to: {}".format(example_plot_path))
        except ImportError:
            print("matplotlib is not installed; skip prediction plot.")
