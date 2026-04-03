import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from ..data_collection.utils import PedestrianStepDataset
from ..models.bc_model import BehaviorCloningPolicy
from ..models.cnn_encoder import CNNEncoder
from ..utils.config_loader import load_config
from ..utils.bc.bc_utils import (
    set_seed,
    move_batch_to_device,
    save_json,
    save_checkpoint,
    build_dataloader,
    split_dataset_by_episode,
    plot_train_val_curves,
    plot_mean_std_curves,
    save_test_summary_csv,
)


EPOCH_PLOT_SPECS = [
    {
        "history_key": "total",
        "filename": "epoch_total_loss.png",
        "title": "Total loss",
        "ylabel": "Loss",
        "ylim": None,
    },
    {
        "history_key": "speed",
        "filename": "epoch_speed_loss.png",
        "title": "Speed loss",
        "ylabel": "Loss",
        "ylim": None,
    },
    {
        "history_key": "direction",
        "filename": "epoch_direction_loss.png",
        "title": "Direction loss",
        "ylabel": "Loss",
        "ylim": None,
    },
    {
        "history_key": "joint_accuracy",
        "filename": "epoch_joint_accuracy.png",
        "title": "Joint accuracy",
        "ylabel": "Accuracy",
        "ylim": (0.0, 1.0),
    },
    {
        "history_key": "speed_accuracy",
        "filename": "epoch_speed_accuracy.png",
        "title": "Speed accuracy",
        "ylabel": "Accuracy",
        "ylim": (0.0, 1.0),
    },
    {
        "history_key": "direction_accuracy",
        "filename": "epoch_direction_accuracy.png",
        "title": "Direction accuracy",
        "ylabel": "Accuracy",
        "ylim": (0.0, 1.0),
    },
    {
        "history_key": "speed_mae",
        "filename": "epoch_speed_mae.png",
        "title": "Speed absolute error",
        "ylabel": "MAE",
        "ylim": None,
    },
    {
        "history_key": "direction_angle_deg",
        "filename": "epoch_direction_angle_deg.png",
        "title": "Direction angle error",
        "ylabel": "Angle (deg)",
        "ylim": None,
    },
]

ITERATION_PLOT_SPECS = [
    {
        "history_key": "total",
        "filename": "iteration_total_loss.png",
        "title": "Training loss",
        "ylabel": "Loss",
    },
    # {
    #     "history_key": "speed",
    #     "filename": "iteration_speed_loss.png",
    #     "title": "Speed loss",
    #     "ylabel": "Loss",
    # },
    # {
    #     "history_key": "direction",
    #     "filename": "iteration_direction_loss.png",
    #     "title": "Direction loss",
    #     "ylabel": "Loss",
    # },
]


def build_model(config, device):
    """Build BC model."""
    bev_feature_dim = config["cnn"]["bev_feature_dim"]
    hidden_dim = config["cnn"]["hidden_dim"]
    direction_dim = config["cnn"]["direction_dim"]

    cnn_encoder = CNNEncoder(input_channels=4, feature_dim=bev_feature_dim)
    model = BehaviorCloningPolicy(
        cnn_encoder=cnn_encoder,
        bev_feature_dim=bev_feature_dim,
        scalar_feature_dim=7,
        hidden_dim=hidden_dim,
        direction_dim=direction_dim,
    ).to(device)

    return model


def normalize_xy_direction(direction_xy, eps=1e-8):
    '''Normalize 2D direction vectors.'''
    norm = torch.norm(direction_xy, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return direction_xy / norm


def masked_direction_loss(pred_direction, gt_direction, direction_mask):
    '''
    Cosine-style direction loss on valid samples only.
    Invalid direction labels (for nearly stationary samples) are ignored.
    '''
    pred_direction = normalize_xy_direction(pred_direction)
    gt_direction = normalize_xy_direction(gt_direction)

    cosine = torch.sum(pred_direction * gt_direction, dim=-1)
    cosine = torch.clamp(cosine, min=-1.0, max=1.0)
    per_sample_loss = 1.0 - cosine

    valid_mask = direction_mask.bool()
    if valid_mask.any():
        return per_sample_loss[valid_mask].mean()

    return pred_direction.sum() * 0.0



def compute_bc_loss(outputs,
                    batch,
                    speed_loss_weight=1.0,
                    direction_loss_weight=1.0):
    '''Compute BC loss using local-frame targets.'''
    gt_speed = batch["target_speed"].unsqueeze(-1)
    gt_direction = batch["target_direction_local"]
    direction_mask = batch["target_direction_mask"]

    pred_speed = outputs["pred_speed"]
    pred_direction = outputs["pred_direction"]

    loss_speed = F.mse_loss(pred_speed, gt_speed)
    loss_direction = masked_direction_loss(
        pred_direction=pred_direction,
        gt_direction=gt_direction,
        direction_mask=direction_mask,
    )
    loss_total = speed_loss_weight * loss_speed + direction_loss_weight * loss_direction

    return {
        "loss_total": float(loss_total.item()),
        "loss_speed": float(loss_speed.item()),
        "loss_direction": float(loss_direction.item()),
    }, loss_total
    

def compute_bc_accuracy(outputs,
                        batch,
                        speed_tolerance=0.20,
                        direction_tolerance_deg=15.0):
    '''
    Compute regression-style accuracy on the new local-frame targets.

    speed_accuracy:
        abs(pred_speed - gt_speed) <= speed_tolerance

    direction_accuracy:
        angle(pred_direction_local, gt_direction_local) <= direction_tolerance_deg
        computed on valid direction samples only

    joint_accuracy:
        for valid direction samples -> both speed and direction must be correct
        for invalid direction samples -> only speed must be correct
    '''
    pred_speed = outputs["pred_speed"].detach().squeeze(-1)
    gt_speed = batch["target_speed"].detach()

    pred_direction = outputs["pred_direction"].detach()
    gt_direction = batch["target_direction_local"].detach()
    direction_mask = batch["target_direction_mask"].detach().bool()

    pred_direction = normalize_xy_direction(pred_direction)
    gt_direction = normalize_xy_direction(gt_direction)

    speed_abs_error = torch.abs(pred_speed - gt_speed)
    speed_correct = speed_abs_error <= speed_tolerance

    dot = torch.sum(pred_direction * gt_direction, dim=-1)
    dot = torch.clamp(dot, min=-1.0, max=1.0)
    direction_angle_deg_all = torch.rad2deg(torch.acos(dot))

    if direction_mask.any():
        direction_angle_deg = float(direction_angle_deg_all[direction_mask].mean().item())
        direction_correct = direction_angle_deg_all <= direction_tolerance_deg
        direction_accuracy = float(direction_correct[direction_mask].float().mean().item())
    else:
        direction_angle_deg = 0.0
        direction_correct = torch.ones_like(speed_correct, dtype=torch.bool)
        direction_accuracy = 0.0

    joint_correct = torch.where(direction_mask, speed_correct & direction_correct, speed_correct)

    return {
        "speed_mae": float(speed_abs_error.mean().item()),
        "direction_angle_deg": direction_angle_deg,
        "speed_accuracy": float(speed_correct.float().mean().item()),
        "direction_accuracy": direction_accuracy,
        "joint_accuracy": float(joint_correct.float().mean().item()),
    }



def create_history(seed, split_seed, speed_tolerance, direction_tolerance_deg):
    '''Create training history dictionary.'''
    epoch_keys = [
        "total",
        "speed",
        "direction",
        "speed_mae",
        "direction_angle_deg",
        "speed_accuracy",
        "direction_accuracy",
        "joint_accuracy",
    ]

    return {
        "seed": seed,
        "split_seed": split_seed,
        "epoch": {
            "train": {key: [] for key in epoch_keys},
            "val": {key: [] for key in epoch_keys},
        },
        "iteration": {
            "total": [],
            "speed": [],
            "direction": [],
        },
        "test": None,
        "split_info": {},
        "accuracy_definition": {
            "type": "tolerance_based_regression_accuracy_local_frame",
            "speed_tolerance": speed_tolerance,
            "direction_tolerance_deg": direction_tolerance_deg,
        },
    }


def append_epoch_metrics(history, stage, metrics):
    '''Append one epoch of metrics to history.'''
    history["epoch"][stage]["total"].append(metrics["loss_total"])
    history["epoch"][stage]["speed"].append(metrics["loss_speed"])
    history["epoch"][stage]["direction"].append(metrics["loss_direction"])
    history["epoch"][stage]["speed_mae"].append(metrics["speed_mae"])
    history["epoch"][stage]["direction_angle_deg"].append(metrics["direction_angle_deg"])
    history["epoch"][stage]["speed_accuracy"].append(metrics["speed_accuracy"])
    history["epoch"][stage]["direction_accuracy"].append(metrics["direction_accuracy"])
    history["epoch"][stage]["joint_accuracy"].append(metrics["joint_accuracy"])


def run_one_epoch(model,
                  loader,
                  device,
                  optimizer=None,
                  speed_tolerance=0.20,
                  direction_tolerance_deg=15.0,
                  speed_loss_weight=1.0,
                  direction_loss_weight=1.0,
                  grad_clip_norm=None,
                  desc=None,
                  history=None):
    '''Run one training or evaluation epoch.'''
    is_training = optimizer is not None

    metric_sums = {
        "loss_total": 0.0,
        "loss_speed": 0.0,
        "loss_direction": 0.0,
        "speed_mae": 0.0,
        "direction_angle_deg": 0.0,
        "speed_accuracy": 0.0,
        "direction_accuracy": 0.0,
        "joint_accuracy": 0.0,
    }
    num_samples = 0

    if is_training:
        model.train()
        loop = tqdm(loader, desc=desc)
    else:
        model.eval()
        loop = loader

    for batch in loop:
        batch = move_batch_to_device(batch, device)

        if is_training:
            outputs = model(batch)
            loss_info, loss = compute_bc_loss(
                outputs=outputs,
                batch=batch,
                speed_loss_weight=speed_loss_weight,
                direction_loss_weight=direction_loss_weight,
            )

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping prevent too large or too small
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(batch)
                loss_info, _ = compute_bc_loss(
                    outputs=outputs,
                    batch=batch,
                    speed_loss_weight=speed_loss_weight,
                    direction_loss_weight=direction_loss_weight,
                )

        with torch.no_grad():
            accuracy_info = compute_bc_accuracy(
                outputs=outputs,
                batch=batch,
                speed_tolerance=speed_tolerance,
                direction_tolerance_deg=direction_tolerance_deg,
            )

        batch_size = int(batch["target_speed"].shape[0])
        num_samples += batch_size

        for key in loss_info:
            metric_sums[key] += loss_info[key] * batch_size
        for key in accuracy_info:
            metric_sums[key] += accuracy_info[key] * batch_size

        if is_training and history is not None:
            history["iteration"]["total"].append(loss_info["loss_total"])
            history["iteration"]["speed"].append(loss_info["loss_speed"])
            history["iteration"]["direction"].append(loss_info["loss_direction"])

            loop.set_postfix({
                "loss": f"{loss_info['loss_total']:.4f}",
                "joint_acc": f"{accuracy_info['joint_accuracy']:.3f}",
            })

    num_samples = max(num_samples, 1)
    return {key: value / num_samples for key, value in metric_sums.items()}


def plot_single_seed_results(history, save_dir, epoch_smooth_window=1, iteration_smooth_window=101):
    '''Save plots for one seed.'''
    os.makedirs(save_dir, exist_ok=True)

    for spec in EPOCH_PLOT_SPECS:
        plot_train_val_curves(
            train_values=history["epoch"]["train"][spec["history_key"]],
            val_values=history["epoch"]["val"][spec["history_key"]],
            title=spec["title"],
            ylabel=spec["ylabel"],
            save_path=os.path.join(save_dir, spec["filename"]),
            smooth_window=epoch_smooth_window,
            use_step_axis=False,
            show_raw=True,
            ylim=spec["ylim"],
        )

    empty_values = np.array([])

    for spec in ITERATION_PLOT_SPECS:
        plot_train_val_curves(
            train_values=history["iteration"][spec["history_key"]],
            val_values=empty_values,
            title=spec["title"],
            ylabel=spec["ylabel"],
            save_path=os.path.join(save_dir, spec["filename"]),
            smooth_window=iteration_smooth_window,
            use_step_axis=True,
            show_raw=True,
            ylim=None,
        )


def plot_multi_seed_results(history_list, save_dir, epoch_smooth_window=1, iteration_smooth_window=101):
    '''Save aggregated plots across multiple seeds.'''
    os.makedirs(save_dir, exist_ok=True)

    for spec in EPOCH_PLOT_SPECS:
        plot_mean_std_curves(
            train_curves=[history["epoch"]["train"][spec["history_key"]] for history in history_list],
            val_curves=[history["epoch"]["val"][spec["history_key"]] for history in history_list],
            title=spec["title"],
            ylabel=spec["ylabel"],
            save_path=os.path.join(save_dir, spec["filename"]),
            smooth_window=epoch_smooth_window,
            use_step_axis=False,
            ylim=spec["ylim"],
        )

    for spec in ITERATION_PLOT_SPECS:
        plot_mean_std_curves(
            train_curves=[history["iteration"][spec["history_key"]] for history in history_list],
            val_curves=[],
            title=spec["title"],
            ylabel=spec["ylabel"],
            save_path=os.path.join(save_dir, spec["filename"]),
            smooth_window=iteration_smooth_window,
            use_step_axis=True,
            ylim=None,
        )


def summarize_test_metrics(history_list, seed_list):
    '''Summarize test metrics across seeds.'''
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

    summary = {
        "seeds": list(seed_list),
        "per_seed": {},
        "mean": {},
        "std": {},
    }

    for seed, history in zip(seed_list, history_list):
        summary["per_seed"][f"seed_{seed}"] = history["test"]

    for metric_name in metric_names:
        values = np.asarray(
            [history["test"][metric_name] for history in history_list],
            dtype=np.float32,
        )
        summary["mean"][metric_name] = float(values.mean())
        summary["std"][metric_name] = float(values.std())

    return summary


def get_seed_dirs(media_root, checkpoint_root, seed):
    """Get output directories for one seed."""
    seed_name = f"seed_{seed}"
    return (
        os.path.join(media_root, seed_name),
        os.path.join(checkpoint_root, seed_name),
    )

# Train for one seed
def train_one_seed(config,
                   train_dataset,
                   val_dataset,
                   test_dataset,
                   split_info,
                   train_seed,
                   split_seed,
                   media_root,
                   checkpoint_root,
                   device):
    """Train one seed and save all outputs."""
    set_seed(train_seed)

    media_dir, checkpoint_dir = get_seed_dirs(media_root, checkpoint_root, train_seed)
    os.makedirs(media_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- get training params' config ---
    params_cfg = config["params"]
    batch_size = params_cfg["batch_size"]
    num_epochs = params_cfg["num_epochs"]
    learning_rate = params_cfg["learning_rate"]
    num_workers = params_cfg["num_workers"]
    weight_decay = params_cfg.get("weight_decay", 1e-4)
    grad_clip_norm = params_cfg.get("grad_clip_norm", 1.0)
    epoch_smooth_window = params_cfg.get("epoch_smooth_window", 1)
    iteration_smooth_window = params_cfg.get("iteration_smooth_window", 101)
    speed_tolerance = params_cfg.get("speed_tolerance", 0.20)
    direction_tolerance_deg = params_cfg.get("direction_tolerance_deg", 15.0)
    speed_loss_weight = params_cfg.get("speed_loss_weight", 1.0)
    direction_loss_weight = params_cfg.get("direction_loss_weight", 1.0)

    # --- load dataset ---
    train_loader = build_dataloader(train_dataset, batch_size, True, num_workers)
    val_loader = build_dataloader(val_dataset, batch_size, False, num_workers)
    test_loader = build_dataloader(test_dataset, batch_size, False, num_workers)

    # --- build model and define optimizer ---
    model = build_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # --- initialize history ---
    history = create_history(
        seed=train_seed,
        split_seed=split_seed,
        speed_tolerance=speed_tolerance,
        direction_tolerance_deg=direction_tolerance_deg,
    )
    history["split_info"] = split_info

    best_val_loss = float("inf")


    # --- training loop ---
    print(f"\n===== Training seed {train_seed} =====")

    for epoch in range(num_epochs):
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            speed_tolerance=speed_tolerance,
            direction_tolerance_deg=direction_tolerance_deg,
            speed_loss_weight=speed_loss_weight,
            direction_loss_weight=direction_loss_weight,
            grad_clip_norm=grad_clip_norm,
            desc=f"Seed {train_seed} | Epoch {epoch + 1}/{num_epochs}",
            history=history,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            speed_tolerance=speed_tolerance,
            direction_tolerance_deg=direction_tolerance_deg,
            speed_loss_weight=speed_loss_weight,
            direction_loss_weight=direction_loss_weight,
        )

        append_epoch_metrics(history, "train", train_metrics)
        append_epoch_metrics(history, "val", val_metrics)

        print(
            f"\nSeed {train_seed} | Epoch [{epoch + 1}/{num_epochs}] "
            f"train_total={train_metrics['loss_total']:.6f}, "
            f"val_total={val_metrics['loss_total']:.6f}, "
            f"train_joint_acc={train_metrics['joint_accuracy']:.4f}, "
            f"val_joint_acc={val_metrics['joint_accuracy']:.4f}"
        )

        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                save_path=os.path.join(checkpoint_dir, "best_model.pt"),
                epoch=epoch + 1,
                extra_info={
                    "seed": train_seed,
                    "val_metrics": val_metrics,
                },
            )

    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best model from: {best_model_path}")

    # --- test model and output results ---
    test_metrics = run_one_epoch(
        model=model,
        loader=test_loader,
        device=device,
        optimizer=None,
        speed_tolerance=speed_tolerance,
        direction_tolerance_deg=direction_tolerance_deg,
        speed_loss_weight=speed_loss_weight,
        direction_loss_weight=direction_loss_weight,
    )
    history["test"] = test_metrics

    print(
        f"Test Result (seed {train_seed}) -> "
        f"test_total={test_metrics['loss_total']:.6f}, "
        f"test_speed={test_metrics['loss_speed']:.6f}, "
        f"test_dir={test_metrics['loss_direction']:.6f}, "
        f"test_joint_acc={test_metrics['joint_accuracy']:.4f}"
    )

    plot_single_seed_results(
        history=history,
        save_dir=media_dir,
        epoch_smooth_window=epoch_smooth_window,
        iteration_smooth_window=iteration_smooth_window,
    )

    save_json(history, os.path.join(checkpoint_dir, "training_history.json"))
    save_json(test_metrics, os.path.join(checkpoint_dir, "test_metrics.json"))

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        save_path=os.path.join(checkpoint_dir, "last_model.pt"),
        epoch=num_epochs,
        extra_info={
            "seed": train_seed,
            "best_val_loss": best_val_loss,
            "test_metrics": test_metrics,
        },
    )

    return history


def train_bc_multi_seed():
    '''Train multiple seeds and save per-seed + aggregated results.'''

    # --- load config ---
    config = load_config("training_config.json")
    dataset_path = config["dataset_path"]

    params_cfg = config["params"]
    val_ratio = params_cfg.get("val_size", 0.15)
    test_ratio = params_cfg.get("test_size", 0.10)
    train_ratio = 1.0 - val_ratio - test_ratio
    split_seed = params_cfg.get("split_seed", 42)
    seed_list = params_cfg.get("seed_list", [1, 2, 3, 4, 5])
    epoch_smooth_window = params_cfg.get("epoch_smooth_window", 1)
    iteration_smooth_window = params_cfg.get("iteration_smooth_window", 101)
    goal_scale = params_cfg.get("goal_scale", 16.0)
    direction_valid_speed_eps = params_cfg.get("direction_valid_speed_eps", 0.05)

    media_root = config.get("media_dir", os.path.join("media", "bc"))
    checkpoint_root = config.get("checkpoint_dir", os.path.join("checkpoints", "bc"))

    os.makedirs(media_root, exist_ok=True)
    os.makedirs(checkpoint_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- load dataset and output train/val/test ---
    dataset = PedestrianStepDataset(
        h5_path=dataset_path,
        use_goal_relative=True,
        goal_scale=goal_scale,
        speed_eps=direction_valid_speed_eps,
    )
    print(f"Total samples: {len(dataset)}")

    train_dataset, val_dataset, test_dataset, split_info = split_dataset_by_episode(
        dataset=dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )

    print(
        f"Episode split -> "
        f"train={split_info['episode']['num_train']}, "
        f"val={split_info['episode']['num_val']}, "
        f"test={split_info['episode']['num_test']}"
    )
    print(
        f"Sample split -> "
        f"train={split_info['sample']['num_train']}, "
        f"val={split_info['sample']['num_val']}, "
        f"test={split_info['sample']['num_test']}"
    )

    history_list = []

    # --- training on multiple seed ---
    for train_seed in seed_list:
        history = train_one_seed(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            split_info=split_info,
            train_seed=train_seed,
            split_seed=split_seed,
            media_root=media_root,
            checkpoint_root=checkpoint_root,
            device=device,
        )
        history_list.append(history)

    # --- plot results ---
    plot_multi_seed_results(
        history_list=history_list,
        save_dir=os.path.join(media_root, "multi_seed"),
        epoch_smooth_window=epoch_smooth_window,
        iteration_smooth_window=iteration_smooth_window,
    )

    summary = summarize_test_metrics(history_list, seed_list)
    summary["split_seed"] = split_seed
    summary["split_info"] = split_info

    save_json(summary, os.path.join(checkpoint_root, "multi_seed_summary.json"))
    save_test_summary_csv(summary, os.path.join(checkpoint_root, "multi_seed_summary.csv"))

    dataset.close()


if __name__ == "__main__":
    train_bc_multi_seed()
