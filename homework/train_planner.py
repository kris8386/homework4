"""
Usage:
    python3 -m homework.train_planner
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric


def train(
    model_name="mlp_planner",
    transform_pipeline=None,
    num_workers=4,
    lr=1e-3,
    batch_size=128,
    num_epoch=40,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training {model_name} on device: {device}")

    # Choose transform pipeline based on model
    if transform_pipeline is None:
        if model_name == "cnn_planner":
            transform_pipeline = "default"
        elif model_name == "transformer_planner":
            transform_pipeline = "aug"
        else:
            transform_pipeline = "state_only"

    data_root = "drive_data"
    train_loader = load_data(
        dataset_path=f"{data_root}/train",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = load_data(
        dataset_path=f"{data_root}/val",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    print(f"Loaded {len(train_loader.dataset)} training samples")
    print(f"Loaded {len(val_loader.dataset)} validation samples")

    # Load model
    model = load_model(
        model_name,
        n_track=10,
        n_waypoints=3,
        d_model=64,
        nhead=4,
        num_layers=2,
    ).to(device)

    # Loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            if model_name == "cnn_planner":
                image = batch["image"].to(device)
                waypoints = batch["waypoints"].to(device)
                mask = batch["waypoints_mask"].to(device)
                preds = model(image)
            else:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                mask = batch["waypoints_mask"].to(device)
                preds = model(track_left, track_right)

            loss = criterion(preds * mask[..., None], waypoints * mask[..., None])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"[Epoch {epoch + 1}/{num_epoch}] Train Loss: {running_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        metric = PlannerMetric()
        with torch.no_grad():
            for batch in val_loader:
                if model_name == "cnn_planner":
                    image = batch["image"].to(device)
                    waypoints = batch["waypoints"].to(device)
                    mask = batch["waypoints_mask"].to(device)
                    preds = model(image)
                else:
                    track_left = batch["track_left"].to(device)
                    track_right = batch["track_right"].to(device)
                    waypoints = batch["waypoints"].to(device)
                    mask = batch["waypoints_mask"].to(device)
                    preds = model(track_left, track_right)

                metric.add(preds, waypoints, mask)

        results = metric.compute()
        val_loss = results["l1_error"]
        print(f"    ➤ Val L1: {val_loss:.4f}, "
              f"Longitudinal: {results['longitudinal_error']:.4f}, "
              f"Lateral: {results['lateral_error']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            print("✅ New best model found and saved (val loss improved)")

        scheduler.step()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        save_model(model)
        print(f"✅ Saved best model with val loss {best_val_loss:.4f}")


if __name__ == "__main__":
    for model_name in ["mlp_planner", "transformer_planner", "cnn_planner"]:
        for lr in [1e-2, 1e-3, 1e-4]:
            print(f"\n=== Training model={model_name}, {lr=}, ===\n")
            train(
                model_name=model_name,
                lr=lr,
                batch_size=128,
                num_epoch=40,
            )
        print(f"✅ Finished training {model_name} with lr={lr}")