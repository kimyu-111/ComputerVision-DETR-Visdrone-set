import os
import requests
import zipfile
import torch
import torch.nn as nn
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import autocast, GradScaler # [추가] 혼합 정밀도용

# =========================================================
# ⚙️ 설정 (Hyperparameters) - 배치 32 효과 내기!
# =========================================================
EPOCHS = 300            # 추가 학습 횟수
LOAD_DIR = "./detr-visdrone-best" # 이어서 학습할 모델 경로
SAVE_DIR = "./detr-visdrone-final" # 최종 저장 경로

# [핵심 설정]
PHYSICAL_BATCH_SIZE = 8   # GPU에 실제로 들어가는 양 (4080 안전빵)
TARGET_BATCH_SIZE = 32    # 우리가 원하는 학습 효과 (배치 32)
ACCUMULATION_STEPS = TARGET_BATCH_SIZE // PHYSICAL_BATCH_SIZE # 32 / 8 = 4번 모아서 쏨

NUM_WORKERS = 4         
LEARNING_RATE = 1e-6    # 미세 조정을 위해 낮춤 (이미 똑똑해졌으니까)
DATA_DIR = "./visdrone_data"

# =========================================================
# 1. 데이터 준비 & Dataset
# =========================================================
def prepare_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    pass 

VISDRONE_CLASSES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
id_map = {i + 1: i for i in range(len(VISDRONE_CLASSES))}
# [수정] 저장된 프로세서 불러오기
try:
    processor = DetrImageProcessor.from_pretrained(LOAD_DIR)
except:
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

class VisDroneDataset(Dataset):
    def __init__(self, img_dir, label_dir, processor):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.processor = processor
        self.target_size = (800, 800)
        self.resize = transforms.Resize(self.target_size)
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        try:
            file_name = self.img_files[idx]
            img_id = file_name.replace('.jpg', '')
            image = Image.open(os.path.join(self.img_dir, file_name)).convert("RGB")
            
            w_orig, h_orig = image.size
            image = self.resize(image)
            w_new, h_new = self.target_size
            
            scale_w = w_new / w_orig
            scale_h = h_new / h_orig

        except:
            return self.__getitem__((idx + 1) % len(self))

        boxes, labels, areas = [], [], []
        label_path = os.path.join(self.label_dir, img_id + '.txt')
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = list(map(float, line.strip().replace(',', ' ').split()))
                    if len(data) > 5 and int(data[5]) in id_map:
                        x, y, w, h = data[0], data[1], data[2], data[3]
                        x *= scale_w
                        y *= scale_h
                        w *= scale_w
                        h *= scale_h
                        
                        if w > 0 and h > 0:
                            boxes.append([x, y, w, h])
                            labels.append(id_map[int(data[5])])
                            areas.append(w * h)
        
        if not boxes:
            boxes, labels, areas = [[0.0, 0.0, 0.0, 0.0]], [0], [0.0]
            
        target = {
            "image_id": idx,
            "annotations": [{"bbox": b, "category_id": l, "area": a, "iscrowd": 0} for b, l, a in zip(boxes, labels, areas)]
        }
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        return {"pixel_values": encoding["pixel_values"].squeeze(), "labels": encoding["labels"][0]}

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": [item["labels"] for item in batch]
    }

def find_dirs(base_path, target_folder):
    for root, dirs, files in os.walk(base_path):
        if "images" in dirs and target_folder in root:
            img_path = os.path.join(root, "images")
            lbl_path = os.path.join(root, "annotations") if "annotations" in dirs else os.path.join(root, "labels")
            return img_path, lbl_path
    return None, None

def plot_loss_graph(train_log, val_log, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_log, label='Train Loss', color='blue')
    plt.plot(val_log, label='Val Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/loss_graph.png")
    print(f"📊 Loss 그래프 저장됨: {save_path}/loss_graph.png")

def generate_heatmap(model, loader, device, save_path):
    print("🧩 혼동 행렬(Heatmap) 생성 중... (시간이 좀 걸립니다)")
    model.eval()
    y_pred = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating for Heatmap"):
            pixel_values = batch["pixel_values"].to(device)
            outputs = model(pixel_values=pixel_values)
            target_sizes = torch.tensor([[800, 800]] * len(pixel_values)).to(device)
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)

            for result in results:
                pred_classes = result['labels'].cpu().numpy()
                y_pred.extend(pred_classes)

    plt.figure(figsize=(12, 8))
    if len(y_pred) > 0:
        sns.countplot(x=y_pred)
        plt.xticks(ticks=range(len(VISDRONE_CLASSES)), labels=VISDRONE_CLASSES, rotation=45)
        plt.title("Predicted Object Distribution")
        plt.savefig(f"{save_path}/prediction_heatmap.png")
        print(f"🔥 예측 분포 히트맵 저장됨: {save_path}/prediction_heatmap.png")
    else:
        print("⚠️ 탐지된 객체가 없어서 히트맵을 그릴 수 없습니다.")

# =========================================================
# 3. 메인 실행부
# =========================================================
if __name__ == '__main__':
    prepare_data()
    TRAIN_IMG, TRAIN_LBL = find_dirs(DATA_DIR, "train")
    VAL_IMG, VAL_LBL = find_dirs(DATA_DIR, "val")
    
    if not TRAIN_IMG: 
        TRAIN_IMG, TRAIN_LBL = find_dirs(DATA_DIR, "VisDrone2019-DET-train")
        VAL_IMG, VAL_LBL = find_dirs(DATA_DIR, "VisDrone2019-DET-val")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 학습 시작 (GPU: {torch.cuda.get_device_name(0)})")
    print(f"🎯 목표 배치 사이즈: {TARGET_BATCH_SIZE} (물리적 배치: {PHYSICAL_BATCH_SIZE} x 누적: {ACCUMULATION_STEPS})")
    
    train_ds = VisDroneDataset(TRAIN_IMG, TRAIN_LBL, processor)
    val_ds = VisDroneDataset(VAL_IMG, VAL_LBL, processor)
    
    train_loader = DataLoader(train_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    # [수정] 이어서 학습하기 위해 저장된 모델 로드
    print(f"📂 모델 불러오는 중... ({LOAD_DIR})")
    try:
        model = DetrForObjectDetection.from_pretrained(LOAD_DIR, ignore_mismatched_sizes=True).to(device)
    except:
        print("⚠️ 저장된 모델을 찾을 수 없어 처음부터 시작합니다.")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", num_labels=10, ignore_mismatched_sizes=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler() # [추가] FP16 스케일러

    train_loss_history = []
    val_loss_history = []
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print(f"🚀 추가 학습 시작! (+{EPOCHS} Epoch)")
    
    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0
        optimizer.zero_grad() # 시작 전 초기화
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for i, batch in enumerate(loop):
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            
            # [추가] Mixed Precision (FP16) 적용
            with autocast():
                outputs = model(pixel_values=pixel_values, labels=labels)
                # Loss를 누적 횟수로 나눠줍니다 (평균을 맞추기 위해)
                loss = outputs.loss / ACCUMULATION_STEPS
            
            # [추가] Scaler로 역전파
            scaler.scale(loss).backward()
            
            # [핵심] 정해진 횟수(4번)마다 업데이트
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # 기록용 Loss는 다시 곱해서 복원
            train_loss += loss.item() * ACCUMULATION_STEPS
            loop.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)
        
        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss) 

        # --- Val ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                pixel_values = batch["pixel_values"].to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss) 
        
        print(f"📊 Epoch {epoch+1} 완료 | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        # --- 무조건 저장 ---
        print(f"💾 모델 저장 중... ({SAVE_DIR})")
        model.save_pretrained(SAVE_DIR)
        processor.save_pretrained(SAVE_DIR)
    
    print("\n🎉 모든 추가 학습 종료!")
    
    plot_loss_graph(train_loss_history, val_loss_history, SAVE_DIR)
    
    model = DetrForObjectDetection.from_pretrained(SAVE_DIR).to(device)
    generate_heatmap(model, val_loader, device, SAVE_DIR)
    
    print(f"✅ 결과 저장 완료: {SAVE_DIR}")