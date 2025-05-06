from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from ultralytics import YOLO


def get_model_RCNN(num_classes, freeze_backbone) -> FasterRCNN:
    """Return Faster R-CNN model with ResNet50 backbone"""
    # Load pre-trained Faster R-CNN
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if freeze_backbone:
        # Freeze all layers except the head
        for param in model.parameters():
            param.requires_grad = False
        for param in model.roi_heads.box_predictor.parameters():
            param.requires_grad = True

    return model


def get_YOLO_model(config, custom_model_path=False) -> YOLO:
    if custom_model_path:
        model = YOLO(custom_model_path)
    else:
        model = YOLO(config["model_type"])
    return model
