import os
import subprocess

if __name__ == "__main__":
    url = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"

    download_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")

    os.makedirs(download_dir, exist_ok=True)
    subprocess.run(["wget", url, "-P", download_dir], check=True)
