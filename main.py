import argparse

from mlflow.tracking import MlflowClient

from src.utils.run import RCNNTrain, yoloTrain

if __name__ == "__main__":
    flavours = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]

    parser = argparse.ArgumentParser(description="Object Detection Inference")
    parser.add_argument(
        "--model", type=str, required=True, help="Choose YOLO or RCNN model"
    )
    parser.add_argument(
        "-f",
        "--flavour",
        choices=flavours,
        required=False,
        default="yolov8n.pt",
        help="Choose YOLO flavour",
    )

    args = parser.parse_args()
    model_type = args.model.upper()
    client = MlflowClient()
    if model_type:
        if model_type == "RCNN":
            RCNNTrain(config_file="configuration/config_RCNN.yaml", client=client)

        if model_type == "YOLO":
            yoloTrain(
                config_file="configuration/config_yolo.yaml",
                flavour=args.flavour,
                client=client,
            )
        else:
            print(f"wrong model chosen {model_type}. Please choose either YOLO or RCNN")
