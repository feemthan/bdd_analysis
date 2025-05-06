from typing import Any

import mlflow
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from src.utils.metrics import compute_precision_recall_iou


def train_yolo_model(
    data_yaml_path, config, model, project_name, custom_model_path=None
) -> tuple:
    """Train YOLOv8 model with the given configuration"""
    # Create a new model from the YOLOv8 pretrained model

    # Set up training configuration
    results = model.train(
        data=data_yaml_path,
        epochs=config["num_epochs"],
        imgsz=config["img_dim"],
        batch=config["batch_size"],
        workers=config["num_workers"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        pretrained=True,
        project=project_name,
        name="train",
        save=True,
        verbose=True,
    )

    # Return final results
    return results, model


def train_one_epoch(model, optimizer, data_loader, device) -> Any | float:
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    loop = tqdm(data_loader, desc="Training", leave=False)
    for _, (images, targets) in enumerate(loop):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)


def validation(model, data_loader, device) -> Any | float:
    """Train the model for one epoch"""
    model.eval()
    metric = MeanAveragePrecision()
    total_loss = 0
    loop = tqdm(data_loader, desc="Validation", leave=False)

    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for _, (images, targets) in enumerate(loop):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            all_outputs.extend(outputs)
            all_targets.extend(targets)
            precision, recall, mean_iou = compute_precision_recall_iou(
                all_outputs, all_targets
            )
            mlflow.log_metrics(
                {
                    "precision": precision,
                    "recall": recall,
                    "iou": mean_iou,
                }
            )

            for i in range(len(outputs)):
                if "scores" in outputs[i]:
                    outputs[i]["scores"] = torch.ones(
                        len((outputs[i]["boxes"])), device=device
                    )
            metric.update(outputs, targets)

            # Forward pass
            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            model.eval()

            total_loss += losses.item()
        result = metric.compute()
        mlflow.log_metrics(
            {
                "map": result["map"].item(),
                "map_50": result["map_50"].item(),
                "map_75": result["map_75"].item(),
            }
        )

    return total_loss / len(data_loader), precision, recall, mean_iou
