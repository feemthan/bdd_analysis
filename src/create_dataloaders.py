import json
import os
from typing import Any

import torch
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Subset
from torchvision import transforms

# from torchmetrics.detection.iou import IntersectionOverUnion
from torchvision.ops import box_iou

load_dotenv()

class_to_idx = {
    "pedestrian": 1,
    "rider": 2,
    "car": 3,
    "truck": 4,
    "bus": 5,
    "train": 6,
    "motorcycle": 7,
    "bicycle": 8,
    "traffic light": 9,
    "traffic sign": 10,
}


def collate_fn(batch) -> tuple[tuple[Any, ...], ...]:
    """Custom collate function for object detection data"""
    return tuple(zip(*batch, strict=False))


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, img_dir, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

        # Load annotations
        with open(ann_file, "r") as f:
            self.annotations = json.load(f)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx) -> tuple[Any, Any]:
        # Load image
        img_name = self.annotations[idx]["name"]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Get annotations
        boxes = []
        labels = []

        for label in self.annotations[idx]["labels"]:
            if "box2d" in label and label["category"] in class_to_idx:
                x1 = label["box2d"]["x1"]
                y1 = label["box2d"]["y1"]
                x2 = label["box2d"]["x2"]
                y2 = label["box2d"]["y2"]

                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([x1, y1, x2, y2])
                labels.append(class_to_idx[label["category"]])

        if len(boxes) == 0:
            boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
            labels = torch.tensor([0], dtype=torch.int64)
        else:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64)

        # Create target dict
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        # Apply transforms
        if self.transforms:
            image = self.transforms(image)

        return image, target


def get_transform(img_dim) -> transforms.Compose:
    """Return data transformations for training"""
    return transforms.Compose(
        [
            transforms.Resize((img_dim, img_dim)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transform(img_dim) -> transforms.Compose:
    """Return data transformations for validation"""
    return transforms.Compose(
        [
            transforms.Resize((img_dim, img_dim)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def compute_precision_recall_iou(
    outputs, targets, iou_threshold=0.5
) -> tuple[float, float, float]:
    tp = 0
    fp = 0
    fn = 0
    total_iou = []

    for pred, target in zip(outputs, targets, strict=False):
        pred_boxes = pred["boxes"]
        # pred_scores = pred["scores"]
        pred_labels = pred["labels"]

        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        if pred_boxes.nelement() == 0 or gt_boxes.nelement() == 0:
            fn += len(gt_boxes)
            fp += len(pred_boxes)
            continue

        ious = box_iou(pred_boxes, gt_boxes)

        for i in range(len(pred_boxes)):
            max_iou, max_idx = ious[i].max(0)
            if max_iou > iou_threshold and pred_labels[i] == gt_labels[max_idx]:
                tp += 1
                total_iou.append(max_iou.item())
            else:
                fp += 1

        fn += len(gt_boxes) - tp

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    mean_iou = sum(total_iou) / len(total_iou) if total_iou else 0.0

    return precision, recall, mean_iou


if __name__ == "__main__":
    import os

    BASE_PATH = os.getenv("BASE_PATH")
    IMG_PATH = os.getenv("IMG_PATH")
    LABELS_PATH = os.getenv("LABELS_PATH")
    TRAIN_LABELS_JSON = os.getenv("TRAIN_LABELS_JSON")
    VAL_LABELS_JSON = os.getenv("VAL_LABELS_JSON")
    # train_dataloader = CustomDataset(
    # TRAIN_LABELS_JSON,
    # f'{IMG_PATH}/train')
    portion = 100
    val_dataloader = CustomDataset(VAL_LABELS_JSON, f"{IMG_PATH}/val")
    subset = Subset(val_dataloader, indices=range(portion))
    print(val_dataloader.__getitem__(0))
