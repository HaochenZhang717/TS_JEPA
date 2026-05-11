"""
    Script to run the short-term forecasting task.
    ---
        We consider the horizon and then predict a single value.
"""

from config.config_downstream import config

import torch
import json
import copy
import logging
import argparse
import pickle
import os

from main.utils import prepare_args
from main.utils import mse, mae, _reduce

import numpy as np
import random

from src.data_loaders.data_loader import get_jepa_loaders, get_evaluation_loaders
from src.models.encoder import Encoder
from src.models.decoder import LinearDecoder

import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # Parse the args and get the config setup
    config = prepare_args(config)

    # Define some parameters
    num_epochs = 500
    context = 32
    num_patches = 10

    # Load device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    # Init Encoder, Decoder, Optimizer

    # Load Data
    print("Load data")
    config["path_data"] = "./data/" + config["data"] + "/" + config["data"] + ".csv"

    loader = get_evaluation_loaders(
        config["path_data"],
        config["batch_size"],
        config["ratio_patches"],
        config["mask_ratio"],
    )

    input_dim = len(loader.dataset[0][0][0])
    # Encoder
    encoder = Encoder(
        num_patches=len(loader.dataset[0][0]),
        dim_in=input_dim,
        kernel_size=config["pretrain_encoder_kernel_size"],
        embed_dim=config["pretrain_encoder_embed_dim"],
        embed_bias=config["pretrain_encoder_embed_bias"],
        nhead=config["pretrain_encoder_nhead"],
        num_layers=config["pretrain_encoder_num_layers"],
        jepa=True,
    )

    decoder = LinearDecoder(emb_dim=config["pretrain_encoder_embed_dim"], patch_size=32)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the pretrained model
    # path_name = "lr_" + str(config["lr_pretrain"]) \
    #         + "_encoder_" + str(config["pretrain_encoder_embed_dim"]) + "_" \
    #         + str(config["pretrain_encoder_nhead"]) + "_" \
    #         + str(config["pretrain_encoder_num_layers"]) \
    #         + "_epoch_" + str(config["checkpoint_to_use"])

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

    checkpoint_path = config.get("checkpoint_path", "")
    if checkpoint_path:
        ckpt_to_load = checkpoint_path
    else:
        ckpt_to_load = config["path_save"] + path_name + ".pt"

    name_loader = torch.load(ckpt_to_load, map_location=device)["encoder"]
    encoder.load_state_dict(name_loader)
    print("Model loaded from: {}".format(ckpt_to_load))

    # We consider training only the decoder head
    param_groups = [{"params": (p for n, p in decoder.named_parameters())}]

    optimizer = torch.optim.AdamW(param_groups, lr=config["lr"])

    # We train the model on the train set
    print("start train")
    for epoch in range(num_epochs):
        encoder.eval()
        decoder.train()
        total_loss = 0
        for context_patches, target_patch in loader:
            context_patches = context_patches.to(device)
            target_patch = target_patch.to(device)
            optimizer.zero_grad()
            encoded_patches = encoder(context_patches)
            summed_embedding = torch.sum(encoded_patches, dim=1)
            predicted_next_patch = decoder(summed_embedding)
            loss = torch.nn.functional.mse_loss(
                predicted_next_patch, target_patch, reduction="mean"
            )

            loss.backward()
            optimizer.step()

            total_loss += loss / config["batch_size"]
        # if epoch % 10 == 0:
        print("Epoch: {} - Total loss: {}".format(epoch, total_loss))

    # We test the model on the last prediction
    # We define the number of steps we will have
    num_steps = (len(loader.dataset.test_df[context * num_patches :])) // context

    predictions = []
    targets = []
    contexts = []
    naive_predictions = []
    total_diff = 0
    l_val_mse = []
    l_val_mae = []
    l_naive_mse = []
    l_naive_mae = []

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for step in range(num_steps):
            # Encode the current context patches
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

            encoded_patches = encoder(current_context)

            # Sum the embeddings of the context patches
            summed_embedding = torch.sum(encoded_patches, dim=1)

            # Predict the next patch using the decoder
            predicted_next_patch = decoder(summed_embedding)

            y_pred = predicted_next_patch.flatten().detach().cpu().numpy()
            y_true = target_value.detach().cpu().numpy()
            y_naive = current_context[:, -1, :].flatten().detach().cpu().numpy()

            # Compute the Loss
            val_mse = mse(y_pred, y_true)
            val_mae = mae(y_pred, y_true)
            naive_mse = mse(y_naive, y_true)
            naive_mae = mae(y_naive, y_true)

            l_val_mse.append(val_mse)
            l_val_mae.append(val_mae)
            l_naive_mse.append(naive_mse)
            l_naive_mae.append(naive_mae)
            predictions.append(y_pred)
            targets.append(y_true)
            contexts.append(current_context.flatten().detach().cpu().numpy())
            naive_predictions.append(y_naive)

    print("MSE Loss is: {}".format(np.mean(l_val_mse)))
    print("MAE Loss is: {}".format(np.mean(l_val_mae)))
    print("Naive baseline MSE Loss is: {}".format(np.mean(l_naive_mse)))
    print("Naive baseline MAE Loss is: {}".format(np.mean(l_naive_mae)))

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
                "Forecast comparison | lr_pretrain={} ema={} checkpoint={}".format(
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
                ax.plot(
                    context_x,
                    context_tail,
                    color="0.55",
                    linewidth=1.2,
                    label="Context",
                )
                ax.plot(
                    future_x,
                    targets[idx],
                    color="#1f77b4",
                    linewidth=2,
                    label="Target",
                )
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
                "Multiple forecast samples | lr_pretrain={} ema={} checkpoint={}".format(
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
