#  DETR for VisDrone Object Detection

#Introduction
This project implements DETR (Detection Transformer)for object detection on the VisDrone dataset. 
VisDrone is a challenging dataset captured by drones, characterized by small objects, high density, and frequent occlusion. 
The goal of this project is to apply the Transformer-based end-to-end detection architecture to aerial imagery, eliminating the need for complex post-processing steps like NMS or anchor generation.

#Key Features
- Transformer-based Detection: Utilizes Attention mechanisms for object detection, replacing traditional CNN-based anchor boxes.
- End-to-End Learning: Streamlined pipeline using Bipartite Matching Loss (Hungarian Loss), removing the need for NMS (Non-Maximum Suppression).
- Optimized for Aerial Imagery: Custom data loader and preprocessing pipeline designed for the VisDrone dataset's unique characteristics (small & dense objects).

#Tech Stack
Language: Python 3.x

Deep Learning Framework: PyTorch, Torchvision

Model Architecture: DETR (ResNet50 Backbone + Transformer Encoder-Decoder)

Libraries: OpenCV, NumPy, COCO API

#Dataset Info
Dataset: VisDrone-DET

Classes: 10 classes (Pedestrian, People, Bicycle, Car, Van, Truck, Tricycle, Awning-tricycle, Bus, Motor)

Characteristics: Different altitudes, camera angles, and lighting conditions.
<img width="1000" height="600" alt="KakaoTalk_20251130_152830438" src="https://github.com/user-attachments/assets/b15b2932-ff84-45e6-a243-8f585cf42fb4" />
<img width="1600" height="1000" alt="KakaoTalk_20251130_160338372_01" src="https://github.com/user-attachments/assets/167b3b91-1f92-43ff-ac48-e1cf62f9c74b" />
<img width="1600" height="1000" alt="KakaoTalk_20251130_160338372_03" src="https://github.com/user-attachments/assets/54aac5ca-8f5b-4ddc-85c5-8884db46e925" />
<img width="1600" height="1000" alt="KakaoTalk_20251130_160338372_08" src="https://github.com/user-attachments/assets/184247e2-0f9c-4b76-9d2e-3e99ee3e2bf4" />

# Experimental Results

### 1. Training Analysis (Loss Curve)
The model was trained for 100 epochs(actually i run about 170 epochs). As shown in the graph below, both training and validation losses converged effectively.
- Trend: The loss decreases steadily, indicating that the DETR model is successfully learning the features of the VisDrone dataset.
- Validation: Although there are some fluctuations in the validation loss (typical for Transformer-based models), the overall trend follows the training loss, demonstrating good generalization without severe overfitting.

![Loss Curve](./assets/loss_curve.png)

# 2. Qualitative Results (Visualization)
The model demonstrates robust detection performance across various scenarios, including different altitudes, lighting conditions, and object densities.

# Daytime & Small Object Detection
In high-altitude aerial shots, the model successfully detects extremely small objects such as pedestrians and distant vehicles. It accurately distinguishes between classes like `Bus` (Pink) and `Car` (Teal).

![Daytime Result](./assets/result_day.jpg)

## Nighttime & Low-light Robustness
Despite low visibility in nighttime scenes, the model effectively detects moving vehicles on the road. This proves the model's robustness against illumination changes.

![Nighttime Result](./assets/result_night.jpg)

# Dense Crowd & Occlusion Handling
In complex intersection scenes with high object density, the model performs well in detecting crowded pedestrians and vehicles. It handles overlapping objects (occlusion) effectively, which is a key challenge in the VisDrone dataset.

![Dense Scene Result](./assets/result_dense.jpg)
