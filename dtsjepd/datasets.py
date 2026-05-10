import pandas as pd
import torch
from torch.utils.data import DataLoader


class TimeSeriesPatchDataset:
    def __init__(self, series, series_split_size=320, patch_size=32, mask_ratio=0.7):
        self.series = series
        self.series_split_size = series_split_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.series) // self.series_split_size

    def __getitem__(self, idx):
        num_splits = len(self.series) // self.series_split_size
        selected_series = self.series[
            (idx % num_splits) * self.series_split_size : (idx % num_splits + 1)
            * self.series_split_size
        ]

        num_patches = len(selected_series) // self.patch_size
        patches = [
            selected_series[i * self.patch_size : (i + 1) * self.patch_size]
            for i in range(num_patches)
        ]
        patches_tensor = torch.stack(patches)

        num_masked = int(num_patches * self.mask_ratio)
        mask_indices = torch.randperm(num_patches)[:num_masked]
        non_mask_indices = torch.tensor(
            [i for i in range(num_patches) if i not in mask_indices.tolist()],
            dtype=torch.long,
        )

        return patches_tensor, mask_indices.long(), non_mask_indices


def resolve_input_cols(df, data_name, input_cols):
    numeric_cols = [
        col for col in df.columns if col != "date" and pd.api.types.is_numeric_dtype(df[col])
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


def load_series_splits(path_data, data_name, input_cols="auto"):
    timestamp_col = "date"
    validation_fraction = 0.05
    test_fraction = 0.3

    df = pd.read_csv(path_data, low_memory=False)
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df.sort_values(by=[timestamp_col], inplace=True)

    selected_cols = resolve_input_cols(df, data_name, input_cols)
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

    train_mean = train_values.mean(0)
    train_std = train_values.std(0).replace(0, 1.0)

    train_tensor = torch.tensor(((train_values - train_mean) / train_std).values).float()
    val_tensor = torch.tensor(((val_values - train_mean) / train_std).values).float()
    test_tensor = torch.tensor(((test_values - train_mean) / train_std).values).float()

    return {
        "train": train_tensor,
        "val": val_tensor,
        "test": test_tensor,
        "selected_cols": selected_cols,
        "train_mean": train_mean.to_dict(),
        "train_std": train_std.to_dict(),
    }


def build_dataloaders(
    path_data,
    data_name,
    batch_size,
    input_cols="auto",
    ratio_patches=10,
    patch_size=32,
    mask_ratio=0.7,
    num_workers=0,
):
    splits = load_series_splits(path_data, data_name, input_cols=input_cols)
    series_split_size = ratio_patches * patch_size

    train_dataset = TimeSeriesPatchDataset(
        splits["train"],
        series_split_size=series_split_size,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
    )
    val_dataset = TimeSeriesPatchDataset(
        splits["val"],
        series_split_size=series_split_size,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
    )
    test_dataset = TimeSeriesPatchDataset(
        splits["test"],
        series_split_size=series_split_size,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, splits


def gather_masked_patches(patches, masks):
    batch_size, _, patch_size, num_channels = patches.shape
    flat = patches.reshape(batch_size, patches.size(1), patch_size * num_channels)
    gather_idx = masks.unsqueeze(-1).repeat(1, 1, flat.size(-1))
    gathered = torch.gather(flat, dim=1, index=gather_idx)
    return gathered.reshape(batch_size, masks.size(1), patch_size, num_channels)
