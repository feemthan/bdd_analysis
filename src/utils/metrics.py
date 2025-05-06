from torchvision.ops import box_iou


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