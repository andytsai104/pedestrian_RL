
import os
import csv
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset


def set_seed(seed: int = 42, reproducible_mode: bool = True):
    """Set random seeds and runtime mode."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make cudnn none deterministic when tuning and deterministic when finalizing
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(reproducible_mode)
        torch.backends.cudnn.benchmark = not bool(reproducible_mode)


def move_batch_to_device(batch, device):
    """Move all tensor values in one batch to the target device."""
    moved_batch = {}
    for key, value in batch.items():
        moved_batch[key] = value.to(device) if torch.is_tensor(value) else value
    return moved_batch


def save_json(data, save_path):
    """Save one dictionary to a json file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Saved: {save_path}")


def save_checkpoint(model, optimizer, save_path, epoch, extra_info=None):
    """Save one model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra_info is not None:
        checkpoint.update(extra_info)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Saved: {save_path}")


def build_dataloader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    persistent_workers=False,
    prefetch_factor=2,
    ):
    """Create one dataloader."""
    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**dataloader_kwargs)



# --- Make sure the episodes are not spliting ---
def _compute_split_counts(total_count, train_ratio, val_ratio, test_ratio):
    '''Compute split counts while keeping the total exact.'''
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = int(total_count * test_ratio)

    remaining = total_count - (train_count + val_count + test_count)

    for _ in range(remaining):
        if train_ratio >= max(val_ratio, test_ratio):
            train_count += 1
        elif val_ratio >= test_ratio:
            val_count += 1
        else:
            test_count += 1

    return train_count, val_count, test_count



def split_dataset_by_episode(dataset, train_ratio, val_ratio, test_ratio, seed=42):
    '''Split dataset by episode name to avoid train / val / test leakage.'''
    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative and train_ratio must be > 0.")

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    episode_names = sorted({episode_name for episode_name, _, _ in dataset.index})
    total_episodes = len(episode_names)

    if total_episodes == 0:
        raise ValueError("No episodes found in dataset.")

    shuffled_episodes = episode_names.copy()
    random.Random(seed).shuffle(shuffled_episodes)

    num_train, num_val, num_test = _compute_split_counts(
        total_count=total_episodes,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    if num_train <= 0:
        raise ValueError("Train split has no episodes. Please adjust the ratios.")

    train_episodes = set(shuffled_episodes[:num_train])
    val_episodes = set(shuffled_episodes[num_train:num_train + num_val])
    test_episodes = set(shuffled_episodes[num_train + num_val:num_train + num_val + num_test])

    train_indices = []
    val_indices = []
    test_indices = []

    for idx, (episode_name, _, _) in enumerate(dataset.index):
        if episode_name in train_episodes:
            train_indices.append(idx)
        elif episode_name in val_episodes:
            val_indices.append(idx)
        elif episode_name in test_episodes:
            test_indices.append(idx)

    split_info = {
        "episode": {
            "train": sorted(list(train_episodes)),
            "val": sorted(list(val_episodes)),
            "test": sorted(list(test_episodes)),
            "num_train": len(train_episodes),
            "num_val": len(val_episodes),
            "num_test": len(test_episodes),
        },
        "sample": {
            "num_train": len(train_indices),
            "num_val": len(val_indices),
            "num_test": len(test_indices),
        },
    }

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
        split_info,
    )




# --- Plotting functions ---
def set_dynamic_x_axis(ax, num_points, use_step_axis=False):
    """
    Dynamic x-axis:
    - if max <= 10  -> ticks every 1
    - if max > 10   -> ticks every 5
    """
    if num_points <= 0:
        return

    if use_step_axis:
        x_max = num_points / 1000.0
        xlabel = r"Step ($\times 10^3$)"
    else:
        x_max = float(num_points)
        xlabel = "Epoch"

    if x_max <= 10:
        tick_step = 1
        x_limit = np.ceil(x_max)
    else:
        tick_step = 5
        x_limit = np.ceil(x_max / tick_step) * tick_step

    ticks = np.arange(0, x_limit + 1e-6, tick_step)

    ax.set_xlim(0, x_limit)
    ax.set_xticks(ticks)
    ax.set_xlabel(xlabel)

def smooth_curve(values, window=1):
    """Smooth one curve with moving average without boundary shrinkage."""
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0 or window <= 1:
        return values

    window = min(window, len(values))

    if window % 2 == 0:
        window += 1
        window = min(window, len(values))
        if window % 2 == 0:
            window -= 1

    if window <= 1:
        return values

    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)

    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def set_plot_style():
    """Set a clean plotting style for paper-like figures."""
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 400,
        "ps.fonttype": 42,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "axes.facecolor": "#f4f4f4",
        "figure.facecolor": "white",
        "grid.color": "#9a9a9a",
        "grid.alpha": 0.35,
        "grid.linewidth": 0.7,
        "lines.linewidth": 2.3,
        "lines.solid_capstyle": "round",
    })


def save_figure(fig, save_path):
    """Save one figure as png."""
    save_root, _ = os.path.splitext(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_root + ".png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_root}.png")


def plot_train_val_curves(train_values,
                          val_values,
                          title,
                          ylabel,
                          save_path,
                          smooth_window=1,
                          use_step_axis=False,
                          show_raw=True,
                          ylim=None):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.1))

    train_values = np.asarray(train_values, dtype=np.float32)
    val_values = np.asarray(val_values, dtype=np.float32)

    x_train = np.arange(len(train_values), dtype=np.float32)
    x_val = np.arange(len(val_values), dtype=np.float32)

    if use_step_axis:
        x_train = x_train / 1000.0
        x_val = x_val / 1000.0

    train_curve = smooth_curve(train_values, smooth_window)
    val_curve = smooth_curve(val_values, smooth_window)

    train_color = "#1f77b4"
    val_color = "#2ca02c"

    if show_raw and smooth_window > 1 and len(train_values) > 1:
        ax.plot(x_train, train_values, color=train_color, alpha=0.18, linewidth=1.0)
    ax.plot(x_train, train_curve, color=train_color, label="Train")

    if len(val_values) > 0:
        if show_raw and smooth_window > 1 and len(val_values) > 1:
            ax.plot(x_val, val_values, color=val_color, alpha=0.16, linewidth=1.0)
        ax.plot(x_val, val_curve, color=val_color, label="Validation")

    ax.set_title(title, pad=6)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=False)
    # Set dynamic x-axis
    set_dynamic_x_axis(ax, len(train_values), use_step_axis=use_step_axis)

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    save_figure(fig, save_path)


def plot_mean_std_curves(train_curves,
                         val_curves,
                         title,
                         ylabel,
                         save_path,
                         smooth_window=1,
                         use_step_axis=False,
                         ylim=None):
    """Plot mean ± std curves across multiple seeds."""
    if len(train_curves) == 0:
        raise ValueError("train_curves must contain at least one run.")

    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.1))

    train_min_length = min(len(curve) for curve in train_curves)
    train_stacked = np.stack(
        [np.asarray(curve[:train_min_length], dtype=np.float32) for curve in train_curves],
        axis=0
    )

    train_mean = smooth_curve(train_stacked.mean(axis=0), smooth_window)
    train_std = smooth_curve(train_stacked.std(axis=0), smooth_window)

    x_train = np.arange(len(train_mean), dtype=np.float32)

    if use_step_axis:
        x_train = x_train / 1000.0
        xlabel = r"Step ($\times 10^3$)"
    else:
        xlabel = "Epoch"

    train_color = "#1f77b4"
    val_color = "#2ca02c"

    ax.plot(x_train, train_mean, color=train_color, label="Train mean")
    ax.fill_between(
        x_train,
        train_mean - train_std,
        train_mean + train_std,
        color=train_color,
        alpha=0.22,
    )

    if len(val_curves) > 0:
        val_min_length = min(len(curve) for curve in val_curves)
        val_stacked = np.stack(
            [np.asarray(curve[:val_min_length], dtype=np.float32) for curve in val_curves],
            axis=0
        )

        val_mean = smooth_curve(val_stacked.mean(axis=0), smooth_window)
        val_std = smooth_curve(val_stacked.std(axis=0), smooth_window)
        x_val = np.arange(len(val_mean), dtype=np.float32)

        if use_step_axis:
            x_val = x_val / 1000.0

        ax.plot(x_val, val_mean, color=val_color, label="Validation mean")
        ax.fill_between(
            x_val,
            val_mean - val_std,
            val_mean + val_std,
            color=val_color,
            alpha=0.18,
        )

    ax.set_title(title, pad=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=False)
    # Set dynamic x-axis
    set_dynamic_x_axis(ax, len(train_mean), use_step_axis=use_step_axis)

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    save_figure(fig, save_path)


def save_test_summary_csv(summary, save_path):
    """Save test result table as csv."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    metric_names = [
        "loss_total",
        "loss_speed",
        "loss_direction",
        "speed_mae",
        "direction_angle_deg",
        "speed_accuracy",
        "direction_accuracy",
        "joint_accuracy",
    ]

    with open(save_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["seed"] + metric_names)

        for seed_name, metric_dict in summary["per_seed"].items():
            writer.writerow([seed_name] + [metric_dict[metric_name] for metric_name in metric_names])

        writer.writerow([])
        writer.writerow(["mean"] + [summary["mean"][metric_name] for metric_name in metric_names])
        writer.writerow(["std"] + [summary["std"][metric_name] for metric_name in metric_names])

    print(f"Saved: {save_path}")
