1. 프로젝트 제목
DETR을 활용한 VisDrone 항공 객체 탐지 (End-to-End Object Detection on VisDrone Dataset using DETR)

2. 프로젝트 한 줄 요약
Transformer 기반의 객체 탐지 모델인 DETR(Detection Transformer)을 구현하여, 객체의 크기가 작고 밀집도가 높은 드론 항공 촬영 영상(VisDrone)에서 효과적으로 객체를 검출하는 프로젝트입니다.

3. 프로젝트 배경 및 목적
배경: 기존의 CNN 기반 탐지 모델(YOLO, Faster R-CNN 등)이 사용하는 Anchor Box나 NMS(비최대 억제)와 같은 복잡한 후처리 과정 없이, Transformer 구조를 사용하여 End-to-End로 객체를 탐지하고자 했습니다.

난이도(Challenge): VisDrone 데이터셋은 드론에서 촬영되어 객체(사람, 차량)가 매우 작고, 밀집되어 있으며, **가려짐(Occlusion)**이 심해 탐지가 매우 어렵습니다.

목적: DETR 모델이 이러한 항공 영상 데이터에서도 유의미한 탐지 성능을 낼 수 있는지 실험하고 학습시키는 것이 목표입니다.

4. 주요 기술 및 특징 (Key Features)
Model Architecture: DETR (ResNet 백본 + Transformer Encoder-Decoder 구조 사용)

Loss Function: 이분 매칭 손실(Hungarian Loss)을 통한 1:1 객체 매칭 학습.

Dataset: VisDrone (다양한 고도와 각도에서 촬영된 차량, 보행자 등 10개 클래스).

Optimization: train.py를 통해 학습 루프 구현, test.py를 통해 성능 평가(mAP 등) 수행.

5. 사용 기술 스택 (Tech Stack)
Language: Python

Framework: PyTorch (Torchvision)

Model: DETR (Detection Transformer)

Tools: OpenCV, NumPy (데이터 전처리 및 시각화)
