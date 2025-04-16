
# Semantic Segmentation Using DeepLabV3 on the COCO Dataset

This repository implements semantic segmentation using DeepLabV3 with a MiT-B2 encoder, trained on a processed subset of the COCO dataset. It involves dataset preparation from COCO annotations and PyTorch Lightning-based training.

## 📦 Task 1: Dataset Preparation (COCO to Masks)

The script data_preprocessing.py converts COCO annotations into segmentation masks.

### ✅ Key Features
- Converts COCO annotations to multi-class masks.
- Handles overlapping objects using per-pixel max logic.
- Skips crowd annotations and invalid entries.
- Processes up to 5,000 images from the COCO dataset.
- Outputs:
  - RGB images to output/images/
  - Grayscale class masks to output/masks/

### ▶ Usage
bash
python data_preprocessing.py \
  --annotations path/to/instances_train2017.json \
  --images path/to/train2017 \
  --output path/to/output_dir \
  --max-images 5000

## 🧠 Task 2: Model Training (DeepLabV3 + MiT-B2)

The training pipeline uses PyTorch Lightning and segmentation_models_pytorch:
- *Architecture*: DeepLabV3+ with MiT-B2 encoder.
- *Loss*: CrossEntropyLoss (with class weights).
- *Metrics*: IoU (Jaccard), Dice Coefficient.
- *Logging*: WandB, mixed precision training.
- *Additional Features*: Early stopping and model checkpointing.

### 🏗 Architecture Highlights
- *DeepLabV3Plus* from segmentation_models_pytorch.
- *Encoder*: mit_b2, pretrained on ImageNet.
- *Image Size*: Resized to 512×512.
- *Augmentations*: Applied using albumentations.

### ▶ Usage
bash
python train.py \
  --data_path path/to/output_dir \
  --epochs 50 \
  --lr 1e-4 \
  --num_classes 80 \
  --batch_size 16 \
  --img_size 512,512


## 🧪 Metrics

- *IoU (Jaccard Index)* per class and overall.
- *Dice Coefficient* averaged across classes.
- *Pixel Accuracy* (optional via WandB logging).

---

## 🧰 Environment Setup

To ensure reproducibility, we use [uv](https://github.com/astral-sh/uv) for environment management.

### ✅ Install dependencies
bash
pip install uv
uv venv
uv pip install -r requirements.txt
---

## 🗃 Project Structure

- data_preprocessing.py — Script to convert COCO annotations to segmentation masks.
- train.py — PyTorch Lightning training script using DeepLabV3+ with MiT-B2 encoder.
- trainning.ipynb — Jupyter notebook for visualizing dataset and predictions.
- requirements.txt — Contains all dependencies to set up the environment using uv.
- results/ — Folder containing sample inputs, ground truth masks, and predicted masks.
  - input1.jpg — Sample input image.
  - mask1.png — Corresponding ground truth mask.
  - pred1.png — Model prediction for the input.
- checkpoints/ — Directory where trained model weights are saved.
- README.md — Project overview, setup instructions, and usage examples.

## ✨ Highlights

- 🔄 *End-to-End Pipeline*: From COCO annotations to trained segmentation model.
- 🖼 *Multi-class Mask Generation*: Converts COCO annotations into pixel-wise labeled masks.
- 💡 *Efficient Model*: Uses DeepLabV3+ with MiT-B2 encoder for strong performance and speed.
- ⚙ *Training Framework*: Built with PyTorch Lightning for modularity and scalability.
- 🎯 *Evaluation Metrics*: Includes IoU, Dice Coefficient, and optional Pixel Accuracy.
- 📊 *WandB Integration*: Logs training metrics and visualizations in real-time.
- 🧪 *Reproducibility*: Uses uv for environment isolation and dependency management.
- 📁 *Well-Structured Codebase*: Easy to modify and extend for new datasets or encoders.
- 🛠 *Robust Preprocessing*: Handles edge cases, invalid entries, and overlapping masks.
- 🔍 *Visualization Ready*: Includes sample inputs, masks, and predictions in results/.
