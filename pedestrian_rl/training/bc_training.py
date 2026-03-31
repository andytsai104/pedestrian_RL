import os
import random
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import json
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from ..data_collection.utils import PedestrianStepDataset
from ..models.bc_model import BehaviorCloningPloicy
from ..models.cnn_encoder import CNNEncoder
from ..utils.config_loader import load_config




def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def compute_bc_loss(outputs, batch, direction_loss_weight=1.0, speed_loss_weight=1.0):
    """
    outputs:
        pred_speed:      (B, 1)
        pred_direction:  (B, 3)

    batch:
        target_speed:      (B,)
        target_direction:  (B, 3)
    """
    gt_speed = batch["target_speed"].unsqueeze(-1)   # (B, 1)
    gt_direction = batch["target_direction"]         # (B, 3)

    pred_speed = outputs["pred_speed"]
    pred_direction = outputs["pred_direction"]

    loss_speed = F.mse_loss(pred_speed, gt_speed)
    loss_direction = F.mse_loss(pred_direction, gt_direction)

    total_loss = speed_loss_weight * loss_speed + direction_loss_weight * loss_direction

    loss_dict = {
        "loss_total": total_loss.item(),
        "loss_speed": loss_speed.item(),
        "loss_direction": loss_direction.item(),
    }
    return total_loss, loss_dict


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_speed_loss = 0.0
    total_direction_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        loss, loss_dict = compute_bc_loss(outputs, batch)

        total_loss += loss_dict["loss_total"]
        total_speed_loss += loss_dict["loss_speed"]
        total_direction_loss += loss_dict["loss_direction"]
        num_batches += 1

    if num_batches == 0:
        return {
            "loss_total": 0.0,
            "loss_speed": 0.0,
            "loss_direction": 0.0,
        }

    return {
        "loss_total": total_loss / num_batches,
        "loss_speed": total_speed_loss / num_batches,
        "loss_direction": total_direction_loss / num_batches,
    }


def plot_results(history, save_dir):
    # Ensure the directory exists once at the start
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(history["train_total"]) + 1)

    # Define the metrics we want to plot
    # Format: (title, train_key, val_key, filename)
    plots = [
        ("Total Loss", "train_total", "val_total", "training_loss.png"),
        ("Speed Loss", "train_speed", "val_speed", "speed_loss.png"),
        ("Direction Loss", "train_direction", "val_direction", "direction_loss.png")
    ]

    for title, train_key, val_key, file_name in plots:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history[train_key], label=f"Train {title}")
        plt.plot(epochs, history[val_key], label=f"Val {title}")
        
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        # Save with 'tight' layout to prevent label clipping
        full_path = os.path.join(save_dir, file_name)
        plt.savefig(full_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {full_path}")




def train_bc():
    set_seed(42)

    # ---------- config ----------
    config_name = "training_config.json"
    config = load_config(config_name=config_name)
    dataset_path = config["dataset_path"]
    save_dir = config["save_dir"]
    figures_save_dir = config["figures_save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    batch_size = config["params"]["batch_size"]
    num_epochs = config["params"]["num_epochs"]
    learning_rate = config["params"]["learning_rate"]
    val_ratio = 0.1
    num_workers = config["params"]["num_workers"]
    bev_feature_dim = config["cnn"]["bev_feature_dim"]
    hidden_dim = config["cnn"]["hidden_dim"]
    direction_dim = config["cnn"]["direction_dim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- dataset ----------
    dataset = PedestrianStepDataset(h5_path=dataset_path, use_goal_relative=True)
    print(f"Total samples: {len(dataset)}")

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # ---------- model ----------
    cnn_encoder = CNNEncoder(input_channels=4, feature_dim=bev_feature_dim)
    model = BehaviorCloningPloicy(
        cnn_encoder=cnn_encoder,
        bev_feature_dim=bev_feature_dim,
        scaler_feature_dim=8,   # velocity(3) + speed(1) + heading(1) + goal_rel(3)
        hidden_dim=hidden_dim,
        direction_dim=direction_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    # ---------- training ----------
    history = {
            "train_total": [],
            "train_speed": [],
            "train_direction": [],
            "val_total": [],
            "val_speed": [],
            "val_direction": [],
    }
    
    for epoch in range(num_epochs):
        model.train()

        running_total = 0.0
        running_speed = 0.0
        running_direction = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            batch = move_batch_to_device(batch, device)

            outputs = model(batch)
            loss, loss_dict = compute_bc_loss(outputs, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_total += loss_dict["loss_total"]
            running_speed += loss_dict["loss_speed"]
            running_direction += loss_dict["loss_direction"]
            num_batches += 1

            progress_bar.set_postfix({
                "loss": f"{loss_dict['loss_total']:.4f}",
                "speed": f"{loss_dict['loss_speed']:.4f}",
                "dir": f"{loss_dict['loss_direction']:.4f}",
            })

        train_metrics = {
            "loss_total": running_total / max(num_batches, 1),
            "loss_speed": running_speed / max(num_batches, 1),
            "loss_direction": running_direction / max(num_batches, 1),
        }

        val_metrics = evaluate(model, val_loader, device)

        print(
            f"\nEpoch [{epoch + 1}/{num_epochs}] "
            f"train_total={train_metrics['loss_total']:.6f}, "
            f"train_speed={train_metrics['loss_speed']:.6f}, "
            f"train_dir={train_metrics['loss_direction']:.6f}, "
            f"val_total={val_metrics['loss_total']:.6f}, "
            f"val_speed={val_metrics['loss_speed']:.6f}, "
            f"val_dir={val_metrics['loss_direction']:.6f}"
        )

        # Keep track on losses
        history["train_total"].append(train_metrics["loss_total"])
        history["train_speed"].append(train_metrics["loss_speed"])
        history["train_direction"].append(train_metrics["loss_direction"])

        history["val_total"].append(val_metrics["loss_total"])
        history["val_speed"].append(val_metrics["loss_speed"])
        history["val_direction"].append(val_metrics["loss_direction"])

        # save best
        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            best_path = os.path.join(save_dir, "best_bc_model.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, best_path)
            print(f"Saved best model to: {best_path}")

    # save plots
    plot_results(
        history=history,
        save_dir=figures_save_dir
    )

    # save history
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    print(f"Saved training history to: {history_path}")

    # save last
    last_path = os.path.join(save_dir, "last_bc_model.pt")
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }, last_path)
    print(f"Saved last model to: {last_path}")


def test_bc_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_name = "training_config.json"
    config = load_config(config_name)

    dataset_path = config["dataset_path"]
    dataset = PedestrianStepDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))

    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)

    encoder = CNNEncoder(input_channels=4, feature_dim=128)
    model = BehaviorCloningPloicy(encoder).to(device)

    outputs = model(batch)
    print(outputs["pred_speed"].shape)      # expected: [4, 1]
    print(outputs["pred_direction"].shape)  # expected: [4, 3]


def test_plotting():    
    history = {
        "train_total": [6, 8, 10],
        "train_speed": [1, 2, 3],
        "train_direction": [5, 6, 7],
        "val_total": [1.7, 2.0, 2.3],
        "val_speed": [1.2, 1.4, 1.6],
        "val_direction": [0.5, 0.6, 0.7],
    }
    config_name = "training_config.json"
    config = load_config(config_name)
    save_dir = config["figures_save_dir"]
    num_epochs = 3
    plot_results(
        history=history,
        save_dir=save_dir
    )


if __name__ == "__main__":
    # test_bc_train()
    # test_plotting()
    train_bc()