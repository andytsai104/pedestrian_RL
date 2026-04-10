import os
import torch
from ..utils.data_utils import PedestrianStepDataset
from ..utils.config_loader import load_config
from ..utils.bc_utils import (
    set_seed,
    get_seed_dirs,
    save_json,
    save_checkpoint,
    build_dataloader,
    split_dataset_by_episode,
    save_test_summary_csv,
    build_model,
    run_one_epoch,
    create_history,
    append_epoch_metrics,
    summarize_test_metrics,
    plot_single_seed_results,
    plot_multi_seed_results
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

    media_dir, checkpoint_dir = get_seed_dirs(media_root, checkpoint_root, train_seed)
    os.makedirs(media_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- get training params' config ---
    params_cfg = config["bc"]["params"]
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

    runtime_cfg = config.get("runtime", {})
    reproducible_mode = runtime_cfg.get("reproducible_mode", True)
    persistent_workers = runtime_cfg.get("persistent_workers", False)
    prefetch_factor = runtime_cfg.get("prefetch_factor", 2)

    # set random seed
    set_seed(train_seed, reproducible_mode=reproducible_mode)

    # --- load dataset ---
    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = build_dataloader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

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
    dataset_path = config["bc"]["dataset_path"]

    params_cfg = config["bc"]["params"]
    val_ratio = params_cfg["val_size"]
    test_ratio = params_cfg["test_size"]
    train_ratio = 1.0 - val_ratio - test_ratio
    split_seed = params_cfg["split_seed"]
    seed_list = params_cfg["seed_list"]
    epoch_smooth_window = params_cfg["epoch_smooth_window"]
    iteration_smooth_window = params_cfg["iteration_smooth_window"]
    goal_scale = params_cfg["goal_scale"]
    clip_bound = params_cfg["clip_bound"]
    future_steps = params_cfg["future_steps"]
    direction_valid_speed_eps = params_cfg["direction_valid_speed_eps"]


    media_root = config["bc"].get("media_dir", os.path.join("media", "bc"))
    checkpoint_root = config["bc"].get("checkpoint_dir", os.path.join("checkpoints", "bc"))
    # media_root = config[""]["media_dir"]
    # checkpoint_root = config["checkpoint_dir"]

    os.makedirs(media_root, exist_ok=True)
    os.makedirs(checkpoint_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- load dataset and output train/val/test ---
    dataset = PedestrianStepDataset(
        h5_path=dataset_path,
        use_goal_relative=True,
        goal_scale=goal_scale,
        clip_bound=clip_bound,
        speed_eps=direction_valid_speed_eps,
        future_steps=future_steps
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
