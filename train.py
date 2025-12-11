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

EPOCHS = 100
BATCH_SIZE = 8
NUM_WORKERS = 4
LEARNING_RATE = 1e-5
SAVE_DIR = "./detr-visdrone-best"
DATA_DIR = "./visdrone_data"

VISDRONE_CLASSES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
id_map = {i + 1: i for i in range(len(VISDRONE_CLASSES))}
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

def prepare_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

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

        boxes, labels = [], []
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
        
        if not boxes:
            boxes, labels = [[0.0, 0.0, 0.0, 0.0]], [0]
            
        target = {
            "image_id": idx,
            "annotations": [{"bbox": b, "category_id": l, "area": 0, "iscrowd": 0} for b, l in zip(boxes, labels)]
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
    print(f"Graph Saved: {save_path}/loss_graph.png")

def generate_heatmap(model, loader, device, save_path):
    print("Generating Heatmap...")
    model.eval()
    y_pred = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
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
        print(f"Heatmap Saved: {save_path}/prediction_heatmap.png")
    else:
        print("No objects detected.")

if __name__ == '__main__':
    prepare_data()
    
    train_img, train_lbl = find_dirs(DATA_DIR, "train")
    val_img, val_lbl = find_dirs(DATA_DIR, "val")
    
    if not train_img: 
        train_img, train_lbl = find_dirs(DATA_DIR, "VisDrone2019-DET-train")
        val_img, val_lbl = find_dirs(DATA_DIR, "VisDrone2019-DET-val")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    train_ds = VisDroneDataset(train_img, train_lbl, processor)
    val_ds = VisDroneDataset(val_img, val_lbl, processor)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", num_labels=10, ignore_mismatched_sizes=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_loss_history = []
    val_loss_history = []
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print(f"Start Training: {EPOCHS} Epochs")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in loop:
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss) 

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss) 
        
        print(f"Epoch {epoch+1} Result | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        model.save_pretrained(SAVE_DIR)
        processor.save_pretrained(SAVE_DIR)
    
    print("Training Finished.")
    plot_loss_graph(train_loss_history, val_loss_history, SAVE_DIR)
    
    model = DetrForObjectDetection.from_pretrained(SAVE_DIR).to(device)
    generate_heatmap(model, val_loader, device, SAVE_DIR)

 