import os

from PIL import Image
from ultralytics import YOLO

# Load a pretrained YOLO11n model
# model = YOLO("yolov8n.pt")
model = YOLO("models/yolo/best.pt")

# Run inference on 'bus.jpg'
# results = model(["https://ultralytics.com/images/bus.jpg", "https://ultralytics.com/images/zidane.jpg"])  # results list

test_dataset_path = "./test"
dir_list = os.listdir(test_dataset_path)[:3]

images_paths = [f"{test_dataset_path}/{i}" for i in dir_list]
results = model(images_paths)  # results list

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Save results to disk
    r.save(filename=f"./output_files/bdd_yolo_test/results_{i}.jpg")
