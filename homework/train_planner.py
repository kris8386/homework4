"""
Usage:
    python3 -m homework.train_planner
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from homework.models import MLPPlanner, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric

def train(
    model_name="mlp_planner",
    transform_pipeline="state_only",
    num_workers=4,
    lr=1e-3,
    batch_size=128,
    num_epoch=40,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset using helper function
    data_root = "drive_data"  # Adjust if needed

    train_loader = load_data(
        dataset_path=f"{data_root}/train",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    val_loader = load_data(
        dataset_path=f"{data_root}/val",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    # Model, Loss, Optimizer
    model = MLPPlanner().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            mask = batch["waypoints_mask"].to(device)

            optimizer.zero_grad()
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
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                mask = batch["waypoints_mask"].to(device)

                preds = model(track_left, track_right)
                metric.add(preds, waypoints, mask)

        results = metric.compute()
        print(f"    ➤ Val L1: {results['l1_error']:.4f}, "
              f"Longitudinal: {results['longitudinal_error']:.4f}, "
              f"Lateral: {results['lateral_error']:.4f}")

    # Save the model
    save_model(model)
    print("✅ Model saved!")


if __name__ == "__main__":
    for lr in [1e-2, 1e-3, 1e-4]:
        train(
            model_name="mlp_planner",
            transform_pipeline="state_only",
            num_workers=4,
            lr=lr,
            batch_size=128,
            num_epoch=40,
        )
