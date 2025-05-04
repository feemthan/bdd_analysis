from typing import Any

import torch.nn as nn
import torchvision.models as models


class Resnet50BBoxRegressor(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super(Resnet50BBoxRegressor, self).__init__()
        # model_location = '/home/feem/.cache/torch/hub/checkpoints'

        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.bbox_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )
        self.cls_head = nn.Linear(in_features, num_classes)

    def forward(self, features) -> tuple[Any, Any]:
        features = self.backbone(features)
        bbox_pred = self.bbox_head(features)
        cls_pred = self.cls_head(features)
        return cls_pred, bbox_pred


class Resnet50BBoxloss(nn.Module):
    def __init__(self) -> None:
        super(Resnet50BBoxloss, self).__init__()
        # model_location = '/home/feem/.cache/torch/hub/checkpoints'
        self.cls_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.SmoothL1Loss()  # nn.MSELoss()

    def forward(self, cls_pred, bbox_pred, cls_target, bbox_target) -> tuple[Any, Any]:
        batch_size = cls_pred.size(0)
        num_objects = cls_target.size(0)

        if batch_size == 1 and num_objects > 1:
            cls_pred = cls_pred.repeat(num_objects, 1)
            bbox_pred = bbox_pred.repeat(num_objects, 1)

        cls_loss = self.cls_loss(cls_pred, cls_target)
        if len(bbox_target.shape) == 3:
            bbox_target = bbox_target.squeeze(1)
        bbox_loss = self.bbox_loss(bbox_pred, bbox_target)
        # bbox_loss2 = self.bbox_loss2(bbox_pred, bbox_target)
        return cls_loss + bbox_loss


if __name__ == "__main__":
    model = Resnet50BBoxRegressor()
    loss = Resnet50BBoxloss()
    print(model)
