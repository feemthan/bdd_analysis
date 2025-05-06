import os

import mlflow
import torch
import torch.optim as optim
import yaml
from src.utils.Dataloaders import (
    CustomDataset,
    collate_fn,
    get_transform,
    get_val_transform,
    prepare_yolo_dataset,
)
from torch.utils.data import DataLoader
from src.utils.trainer import train_one_epoch, train_yolo_model, validation

from src.utils.common import log_model_with_pyproject_env

CONFIG_PATH = "configuration"


def yoloTrain(config_file, client) -> None:
    from src.model.models import get_YOLO_model
    from ultralytics import settings
    settings.update({"mlflow": True})
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment_name = "bdd_detection_yolov8_experiment"
    mlflow.set_experiment(experiment_name)
    mlflow.set_tag("mlflow.runName", "yolov8_training")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    TRAIN_LABELS_JSON = config["TRAIN_LABELS_JSON"]
    VAL_LABELS_JSON = config["VAL_LABELS_JSON"]
    IMG_PATH = config["IMG_PATH"]
    data_yaml_path = os.path.join("dataset_yolo", "data.yaml")
    if not os.path.exists(data_yaml_path):
        data_yaml_path = prepare_yolo_dataset(
            TRAIN_LABELS_JSON, VAL_LABELS_JSON, IMG_PATH
        )
    config_file = os.path.join(CONFIG_PATH, config["custom_model"])
    model = get_YOLO_model(config=config, custom_model_path=config_file)
    results, model = train_yolo_model(
        data_yaml_path=data_yaml_path,
        model=model,
        config=config,
        project_name=config['project_name'],
        custom_model_path=config_file,
    )

    print("the following are the results of the training of YOLO model")
    print(results)


def RCNNTrain(config_file, client) -> None:
    from src.model.models import get_model_RCNN

    experiment_name = "bdd_detection_FRCNN_experiment"
    pip_reqs = log_model_with_pyproject_env()

    mlflow.set_experiment(experiment_name)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Dataset paths
    TRAIN_LABELS_JSON = config["TRAIN_LABELS_JSON"]
    VAL_LABELS_JSON = config["VAL_LABELS_JSON"]
    IMG_PATH = config["IMG_PATH"]

    # Model hyperparameters
    num_classes = config["num_classes"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    subset_size = config["subset_size"]
    num_epochs = config["num_epochs"]
    train_val_split = config["train_val_split"]
    img_dim = config["img_dim"]
    freeze_backbone = config["freeze_backbone"]

    # Create dataset
    train_dataset = CustomDataset(
        ann_file=TRAIN_LABELS_JSON,
        img_dir=f"{IMG_PATH}/train",
        transforms=get_transform(img_dim),
    )

    val_dataset = CustomDataset(
        ann_file=VAL_LABELS_JSON,
        img_dir=f"{IMG_PATH}/val",
        transforms=get_val_transform(img_dim),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(
            {
                "num_classes": num_classes,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "learning_rate": learning_rate,
                "subset_size": subset_size,
                "num_epochs": num_epochs,
                "train_val_split": train_val_split,
                "device": str(device),
                "model_type": "faster_rcnn_resnet50_fpn",
            }
        )

        model = get_model_RCNN(num_classes, freeze_backbone)
        model.to(device)

        # Log trainable parameters summary
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        mlflow.log_params(
            {
                "trainable_params": trainable_params,
                "total_params": total_params,
            }
        )

        # Create optimizer
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            params_to_update, lr=learning_rate, weight_decay=weight_decay
        )
        best_loss = float("inf")
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            avg_loss = train_one_epoch(model, optimizer, train_dataloader, device)
            val_loss, precision, recall, mean_iou = validation(
                model, val_dataloader, device
            )
            mlflow.log_metrics(
                {
                    "train_loss": avg_loss,
                    "val_loss": val_loss,
                    "precision": precision,
                    "recall": recall,
                    "iou": mean_iou,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch+1}/{num_epochs}: Avg Loss: {avg_loss:.4f} Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss

                torch.save(model.state_dict(), "artifacts/best_model.pth")
                # This can also be done for each experiment wise
                # But not done here for the sake of space consumption
                # mlflow.log_artifact("best_model.pth")
                mlflow.log_params(config)
                mlflow.pytorch.log_model(
                    model,
                    artifact_path="pytorch_model",
                    registered_model_name="MyFasterRCNN",
                    pip_requirements=pip_reqs,
                )
                # mlflow.pytorch.log_state_dict(model.state_dict(), artifact_path="model_state_dict")

                print("Training complete! Model saved to 'best_model.pth'")
                if val_loss < 0.01:
                    break
                mlflow.log_metrics(
                    {
                        "best_loss": best_loss,
                    }
                )
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Experiment ID: {mlflow.active_run().info.experiment_id}")
