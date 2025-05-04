# import general dependencies
import os

# import mlflow dependencies
import mlflow

# import torch dependencies
import torch
import torch.optim as optim
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader#, Subset

from src.create_dataloaders import (
    CustomDataset,
    collate_fn,
    get_transform,
    get_val_transform,
)
from src.trainer import get_model, train_one_epoch, validation

load_dotenv()

client = MlflowClient()
print(f"MLFlow Tracking URI: {mlflow.get_tracking_uri()}")

BASE_PATH = os.getenv("BASE_PATH")
IMG_PATH = f"{BASE_PATH}/{os.getenv("IMG_PATH")}"
LABELS_PATH = f"{BASE_PATH}/{os.getenv("LABELS_PATH")}"
TRAIN_LABELS_JSON = f"{BASE_PATH}/{os.getenv("TRAIN_LABELS_JSON")}"
VAL_LABELS_JSON = f"{BASE_PATH}/{os.getenv("VAL_LABELS_JSON")}"

experiment_name = "bdd_detection_frcnn_resnet50_experiment"

try:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    print(f"Using existing experiment '{experiment_name}' with ID: {experiment_id}")
except (mlflow.exceptions.MlflowException, AttributeError) as e:
    print(f"MLflow exception: {e}")

    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")

mlflow.set_experiment(experiment_id=experiment_id)


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load hyperparameters from config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    num_classes = config['num_classes']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    subset_size = config['subset_size']
    num_epochs = config['num_epochs']
    train_val_split = config['train_val_split']
    img_dim = config['img_dim']
    freeze_backbone = config['freeze_backbone']

    # Create dataset
    train_dataset = CustomDataset(
        ann_file=TRAIN_LABELS_JSON, img_dir=f"{IMG_PATH}/train", transforms=get_transform(img_dim)
    )

    val_dataset = CustomDataset(
        ann_file=VAL_LABELS_JSON, img_dir=f"{IMG_PATH}/val", transforms=get_val_transform(img_dim)
    )

    # Use a subset for faster training
    # subset = Subset(val_dataset, indices=range(subset_size))
    # train, val = torch.utils.data.random_split(val_dataset, train_val_split)

    # Create data loaders
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

    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_params({
            "num_classes": num_classes,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "learning_rate": learning_rate,
            "subset_size": subset_size,
            "num_epochs": num_epochs,
            "train_val_split": train_val_split,
            "device": str(device),
            "model_type": "faster_rcnn_resnet50_fpn",
        })
        # Create model
        model = get_model(num_classes, freeze_backbone)
        model.to(device)

        # Log trainable parameters summary
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        mlflow.log_params({
            "trainable_params": trainable_params,
            "total_params": total_params,
        })

        # Create optimizer
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params_to_update, lr=learning_rate, weight_decay=weight_decay)
        # optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Train model
        best_loss = float('inf')
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("-" * 50)

            avg_loss = train_one_epoch(model, optimizer, train_dataloader, device)
            val_loss, precision, recall, mean_iou = validation(model, val_dataloader, device)
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "precision": precision,
                "recall": recall,
                "iou": mean_iou,
            }, step=epoch)

            print(f"Epoch {epoch+1}/{num_epochs}: Avg Loss: {avg_loss:.4f} Val Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss

                torch.save(model.state_dict(), "best_model.pth")
                # This can also be done for each experiment wise
                # But not done here for the sake of space consumption
                mlflow.log_artifact("./artifacts/best_model.pth")
                mlflow.log_artifact("./artifacts/config.yaml")
                print("Training complete! Model saved to 'best_model.pth'")
                if val_loss < 0.01:
                    break

        mlflow.log_metrics({
            "best_loss": best_loss,
        })
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Experiment ID: {mlflow.active_run().info.experiment_id}")
    