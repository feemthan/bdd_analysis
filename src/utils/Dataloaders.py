import json
import os
import shutil
from typing import Any, LiteralString

import cv2
import torch
import yaml
from PIL import Image
from torchvision import transforms

from src.constants import class_to_idx


def collate_fn(batch) -> tuple[tuple[Any, ...], ...]:
    """Custom collate function for object detection data"""
    return tuple(zip(*batch, strict=False))


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, img_dir, transforms=None) -> None:
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


def create_yolo_label_file(annotation, img_dir, output_dir) -> None | Any:
    """
    Create a YOLO format label file for a single image annotation
    """
    img_name = annotation["name"]
    img_path = os.path.join(img_dir, img_name)

    # Get image dimensions (needed for bbox normalization)
    try:
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
    except Exception as e:
        print(f"Error reading image {img_path}: {e}")
        return None

    # Create label filename (same as image but with .txt extension)
    label_filename = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(output_dir, label_filename)

    # Write label file
    with open(label_path, "w") as f:
        for label in annotation["labels"]:
            if "box2d" in label and label["category"] in class_to_idx:
                x1 = label["box2d"]["x1"]
                y1 = label["box2d"]["y1"]
                x2 = label["box2d"]["x2"]
                y2 = label["box2d"]["y2"]

                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                # Convert to YOLO format
                x_center, y_center, width, height = convert_bbox_to_yolo(
                    [x1, y1, x2, y2], img_width, img_height
                )

                # Write in YOLO format: class_idx x_center y_center width height
                class_idx = class_to_idx[label["category"]]
                f.write(f"{class_idx} {x_center} {y_center} {width} {height}\n")

    return img_name


def convert_bbox_to_yolo(bbox, img_width, img_height) -> tuple:
    """
    Convert bounding box from [x1, y1, x2, y2] format to YOLO format [x_center, y_center, width, height]
    All values are normalized between 0 and 1
    """
    x1, y1, x2, y2 = bbox

    # Calculate center, width, height and normalize
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    return x_center, y_center, width, height


def prepare_yolo_dataset(train_json, val_json, img_base_dir) -> LiteralString:
    """
    Prepare dataset for YOLOv8 training
    1. Create dataset directory structure
    2. Convert annotations to YOLO format
    3. Create data.yaml file
    """
    # Create base directory for YOLOv8 dataset
    yolo_dataset_dir = "dataset_yolo"
    os.makedirs(yolo_dataset_dir, exist_ok=True)
    class_names = list(class_to_idx.keys())
    # Create directory structure
    img_train_dir = os.path.join(yolo_dataset_dir, "images", "train")
    img_val_dir = os.path.join(yolo_dataset_dir, "images", "val")
    labels_train_dir = os.path.join(yolo_dataset_dir, "labels", "train")
    labels_val_dir = os.path.join(yolo_dataset_dir, "labels", "val")

    os.makedirs(img_train_dir, exist_ok=True)
    os.makedirs(img_val_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    # Process training data
    with open(train_json, "r") as f:
        train_annotations = json.load(f)

    train_images = []
    for annotation in train_annotations:
        img_name = create_yolo_label_file(
            annotation, f"{img_base_dir}/train", labels_train_dir
        )
        if img_name:
            src_path = os.path.join(f"{img_base_dir}/train", img_name)
            dst_path = os.path.join(img_train_dir, img_name)
            shutil.copy(src_path, dst_path)
            train_images.append(dst_path)

    # Process validation data
    with open(val_json, "r") as f:
        val_annotations = json.load(f)

    val_images = []
    for annotation in val_annotations:
        img_name = create_yolo_label_file(
            annotation, f"{img_base_dir}/val", labels_val_dir
        )
        if img_name:
            src_path = os.path.join(f"{img_base_dir}/val", img_name)
            dst_path = os.path.join(img_val_dir, img_name)
            shutil.copy(src_path, dst_path)
            val_images.append(dst_path)

    # Create data.yaml file
    data_yaml = {
        "path": os.path.abspath(yolo_dataset_dir),
        "train": os.path.abspath(img_train_dir),
        "val": os.path.abspath(img_val_dir),
        "names": class_names,
        "nc": len(class_names),
    }

    data_yaml_path = os.path.join(yolo_dataset_dir, "data.yaml")
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"Dataset prepared for YOLOv8 training at {yolo_dataset_dir}")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")

    return data_yaml_path


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
