import os
import argparse
import json
from typing import Dict, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import yaml
# from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor

from src.constants import class_to_idx
from src.utils.Dataloaders import get_transform

def load_model(model_path: str, num_classes: int) -> FasterRCNN:
    """Load the trained Faster R-CNN model"""
    model = fasterrcnn_resnet50_fpn(weights=None)
    
    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load the trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)

    return model


def prepare_image(image_path: str, img_dim: int) -> torch.Tensor:
    """Prepare image for inference"""
    image = Image.open(image_path).convert("RGB")
    transform = get_transform(img_dim)
    return transform(image).unsqueeze(0)  # Add batch dimension


def get_prediction(model: FasterRCNN, image: torch.Tensor, 
                   device: torch.device, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
    """Run inference on a single image"""
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        prediction = model(image)[0]
        
        # Filter by confidence threshold
        keep_idxs = prediction['scores'] > threshold
        filtered_prediction = {
            'boxes': prediction['boxes'][keep_idxs],
            'labels': prediction['labels'][keep_idxs],
            'scores': prediction['scores'][keep_idxs]
        }
        
    return filtered_prediction


def visualize_predictions(image_path: str, prediction: Dict[str, torch.Tensor], 
                         output_path: Optional[str] = None) -> None:
    """Visualize the predictions on the image"""
    # Create inverse mapping from indices to class names
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    idx_to_class[0] = "background"  # Add background class
    
    # Load the original image
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    # Color map for different classes
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_to_idx) + 1))
    
    # Draw bounding boxes and labels
    for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores'], strict=False):
        x1, y1, x2, y2 = box.cpu().numpy()
        class_name = idx_to_class[label.item()]
        color = colors[label.item()]
        
        # Create rectangle
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label text
        plt.text(x1, y1-5, f"{class_name}: {score:.2f}", 
                 color='white', fontsize=10, 
                 bbox=dict(facecolor=color, alpha=0.7))
    
    plt.title(f"Detections: {len(prediction['boxes'])}")
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def process_directory(model: FasterRCNN, directory: str, output_dir: str, 
                     device: torch.device, img_dim: int, conf_threshold: float = 0.5) -> None:
    """Process all images in a directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in os.listdir(directory) 
                  if os.path.splitext(f)[1].lower() in image_extensions]
    
    results = []
    
    for img_file in image_files:
        img_path = os.path.join(directory, img_file)
        print(f"Processing {img_path}...")
        
        # Prepare image
        image_tensor = prepare_image(img_path, img_dim)
        
        # Get prediction
        prediction = get_prediction(model, image_tensor, device, conf_threshold)
        
        # Convert tensors to lists for saving to JSON
        prediction_json = {
            'image': img_file,
            'boxes': prediction['boxes'].cpu().numpy().tolist(),
            'labels': prediction['labels'].cpu().numpy().tolist(),
            'scores': prediction['scores'].cpu().numpy().tolist()
        }
        results.append(prediction_json)
        
        # Save visualization
        output_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_detection.jpg")
        visualize_predictions(img_path, prediction, output_path)
    
    # Save all results to a JSON file
    with open(os.path.join(output_dir, 'detection_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(image_files)} images. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Object Detection Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    parser.add_argument("--input", type=str, required=True, 
                       help="Path to input image or directory of images")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run inference on (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, config['num_classes'])
    model.to(device)
    print(f"Model loaded from {args.model}")
    
    if os.path.isdir(args.input):
        # Process directory
        process_directory(model, args.input, args.output, device, 
                         config['img_dim'], args.threshold)
    else:
        # Process single image
        os.makedirs(args.output, exist_ok=True)
        
        # Prepare image
        image_tensor = prepare_image(args.input, config['img_dim'])
        
        # Get prediction
        prediction = get_prediction(model, image_tensor, device, args.threshold)
        
        # Save visualization
        output_path = os.path.join(args.output, f"{os.path.basename(args.input).split('.')[0]}_detection.jpg")
        visualize_predictions(args.input, prediction, output_path)
        
        # Save prediction to JSON
        output_json = os.path.join(args.output, f"{os.path.basename(args.input).split('.')[0]}_detection.json")
        with open(output_json, 'w') as f:
            json.dump({
                'image': os.path.basename(args.input),
                'boxes': prediction['boxes'].cpu().numpy().tolist(),
                'labels': prediction['labels'].cpu().numpy().tolist(),
                'scores': prediction['scores'].cpu().numpy().tolist()
            }, f, indent=2)
        
        print(f"Processed image {args.input}. Results saved to {args.output}")


if __name__ == "__main__":
    main()