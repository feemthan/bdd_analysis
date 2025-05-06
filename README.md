# BDD Object Detection Benchmark

![BDD100k Dataset](https://bair.berkeley.edu/blog/assets/scalabel/splash.gif)

> A comparison of Faster R-CNN and YOLOv8 for autonomous driving object detection using the BDD100k dataset.

## 📋 Table of Contents

- [Overview]
- [Project Structure]
- [Models]
- [Installation]
- [Usage]
- [MLflow Tracking]
- [Configuration]
- [Results]
- [Contributing]
- [License]

## 🔍 Overview

This project provides a comprehensive comparison between two popular object detection architectures - Faster R-CNN and YOLOv8 - using the Berkeley Deep Drive (BDD100k) dataset. The BDD100k dataset is one of the largest and most diverse driving datasets, containing over 100,000 videos with rich annotations including object detection, instance segmentation, and more.

The primary goal of this project is to:

- Implement and train both Faster R-CNN and YOLOv8 models on the same dataset
- Evaluate and compare their performance metrics (precision, recall, mAP, IoU)
- Track experiments using MLflow
- Provide a Docker-based environment for reproducible research

Both models are trained to detect 10 key object classes important for autonomous driving scenarios: pedestrians, riders, cars, trucks, buses, trains, motorcycles, bicycles, traffic lights, and traffic signs.

## 🏗️ Project Structure

```bash
.
├── assignment_data_bdd/    # Default data
├── configuration/          # Configuration files
│   ├── config_RCNN.yaml
│   ├── config_yolo.yaml
│   ├── yolov8_custom_model.yaml
├── dataset_yolo/           # Processed YOLO dataset
│   ├── images/
│   ├── ├── train/
│   ├── ├── val/
│   ├── labels
│   ├── ├── train/
│   ├── ├── val/
│   ├── data.yaml
├── mlruns/                 # MLflow runs data
├── models/                 # Models save files
│   ├── rcnn/               # RCNN model save files
│   ├── ├── best_model.pt/  # Best model save
│   ├── yolo/               # YOLO model save files
│   ├── ├── args.yaml/      # best model save files
│   ├── ├── best_model.pt/  # best model save files
│   ├── ├── best.pt/        # best model save files
├── src/
│   ├── model/              # Model definitions
│   ├── ├── models/         # Model definitions
│   ├── utils/              # Utility functions
│   │   ├── Dataloaders.py  # Data loaders and preprocessing
│   │   ├── common.py       # Common utility functions
│   │   ├── logger.py       # Logging setup
│   │   ├── metrics.py      # Evaluation metrics
│   │   ├── run.py          # Training execution
│   │   └── trainer.py      # Training loops
├── download_frcnn.sh       # Main configuration
├── download_yolov8.sh      # Docker services definition
├── config.yaml             # Main configuration
├── docker-compose.yaml     # Docker services definition
├── Dockerfile.mlflow       # MLflow server dockerfile
├── Dockerfile.rcnn         # Faster R-CNN dockerfile
├── Dockerfile.yolo         # YOLOv8 dockerfile
├── main.py                 # Main entry point
├── pyproject.toml          # Python project definition
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

### Faster R-CNN

- Two-stage detector with Region Proposal Network (RPN)
- ResNet50 backbone with Feature Pyramid Network (FPN)
- Higher accuracy but slower inference

### YOLOv8

- Single-stage detector
- Excellent balance of speed and accuracy
- Multiple model sizes (n, s, m, l, x) for different compute requirements

## 🛠️ Installation

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- NVIDIA Container Toolkit

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/feemthan/bdd_analysis.git
   cd bdd_analysis
   ```

2. Download the BDD100k dataset:

   ```bash
   # Create directory for dataset
   unzip assignment_data_bdd.zip
   ```

3. Build Docker containers:

   ```bash
   docker compose down
   docker compose up --build -d
   ```

### Run Faster R-CNN Training

```bash
docker compose run bdd python main.py --model RCNN
```

### Run YOLOv8 Training (default is YOLOv8n all flavors are downloaded)

```bash
docker compose run yolo python main.py --model YOLO
```

### Run YOLOv8 custom Training

```bash
# edit configuration/yolov8_custom_model.yaml && custom_model==True
# This will pick the custom model from the configuration file
# configuration/yolov8_custom_model.yaml
docker compose run yolo python main.py --model YOLO
```

### Interactive Session

```bash
# For Faster R-CNN container
docker-compose run bdd bash

# For YOLOv8 container
docker-compose run yolo bash
```

## 📊 MLflow Tracking

This project uses MLflow to track experiments, metrics, and artifacts. The MLflow UI is accessible at `http://localhost:5000` once the MLflow server is running.

Tracked metrics include:

- Training and validation loss
- Precision and recall
- Mean IoU (Intersection over Union)
- mAP (mean Average Precision) at different IoU thresholds

### Model-Specific Configuration

- For Faster R-CNN: `configuration/config_RCNN.yaml`
- For YOLOv8: `configuration/config_yolo.yaml`

## 📈 Results

### Performance Metrics

| Model        | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Inference Time (ms) |
| ------------ | --------- | ------ | ------- | ------------ | ------------------- |
| Faster R-CNN | TBD       | TBD    | TBD     | TBD          | TBD                 |
| YOLOv8n      | TBD       | TBD    | TBD     | TBD          | 18ms           |

### Visualization

MLflow provides visualization of metrics across training. Visit the MLflow UI at `http://localhost:5000` to view:

- Learning curves
- Precision-recall curves
- Streamlit app for interactive visualization

## Future Improvements

- Streamlit for RCNN
  This was not built due to the model not performing as well as YOLOv8 in this timeframe. RCNN is slower but more accurate. But requires much hiher compute resources.
  By the time of writing this, I had not yet completed the RCNN model inference script. Will add this if time permits.
- More efficient data preprocessing and augmentation
   Not a lot of time was spent on this, this is something that was in the pipeline but unfortunately slipped.
- Hyperparameter tuning in Faster R-CNN
   The tuning was done but not enough time was spent on this. due to the complexity of the dataset and the model. Perhaps a more efficient way to do this was to take a larger subset of the train dataset and train to solve for time.
- Model expansion with adding more layers for both Faster R-CNN and YOLOv8.
  I spent hours thinking about how to implement this more efficiently with partial activation of a few outermost layers or even quantization but I saw that adding one more layer was adding 20% more time to the training (mainly for YOLOv8). Also this project was not to fully showcase the best model but to show the capabilities of the model and to pipeline it well.

## 🙏 Acknowledgements

- [BDD100K Dataset](https://bdd-data.berkeley.edu/)
- [PyTorch](https://pytorch.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Torchvision Object Detection](https://pytorch.org/vision/stable/models.html#object-detection)
- [MLflow](https://mlflow.org/)
