# waste-classification
A deep learning model for classifying waste into 10 categories and detecting multiple objects in a single image, using a custom CNN, task‑tailored metrics and other stuff.

## Scripts
- **`regular_cnn.py`** – Starts with a simple CNN and evolves into a ResNet‑style CNN with residual blocks.
- **`fom.py`** – Full advanced pipeline with task‑tailored FoM and multi‑object detection.

## Features
- 10‑class waste classification
- Residual blocks for deeper training
- Task‑tailored Figure of Merit (FoM) for hazardous item recall
- Class weights and offline data augmentation
- Multi‑object detection using Selective Search, edge‑density filtering, batch inference, and NMS

## Model Architecture
Below is the architecture of the first regular CNN model:

![Model Architecture](reg_cnn.jpg)

## Detailed Project Documentation
For a full explanation of the project, including motivation, methodology, results, and analysis, refer to:

[Project Details Document](project_details.pdf)

## Installation
```bash
pip install -r requirements.txt
