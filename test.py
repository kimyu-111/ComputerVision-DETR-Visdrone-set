import os
import random
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

SAVE_DIR = "./detr-visdrone-best"
DATA_DIR = "./visdrone_data"
THRESHOLD = 0.7

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print(f"Loading Model from {SAVE_DIR}...")
try:
    model = DetrForObjectDetection.from_pretrained(SAVE_DIR).to(device)
    processor = DetrImageProcessor.from_pretrained(SAVE_DIR)
    model.eval()
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"Error Loading Model: {e}")
    exit()

VISDRONE_CLASSES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]
id_to_class = {i: name for i, name in enumerate(VISDRONE_CLASSES)}

def find_image_folder(base_path):
    for root, dirs, files in os.walk(base_path):
        if "images" in dirs:
            path = os.path.join(root, "images")
            if len([f for f in os.listdir(path) if f.endswith('.jpg')]) > 0:
                return path
    return None

target_dir = find_image_folder(DATA_DIR)
if not target_dir:
    print("Image folder not found.")
    exit()

img_files = [f for f in os.listdir(target_dir) if f.endswith('.jpg')]
random_file = random.choice(img_files)
image_path = os.path.join(target_dir, random_file)
print(f"Test Image: {random_file}")

orig_image = Image.open(image_path).convert("RGB")
input_image = orig_image.resize((800, 800))

inputs = processor(images=input_image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

target_sizes = torch.tensor([orig_image.size[::-1]]).to(device)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=THRESHOLD)[0]

plt.figure(figsize=(16, 10))
plt.imshow(orig_image)
ax = plt.gca()
colors = ['red', 'lime', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'pink', 'white']

print(f"Detected Objects: {len(results['scores'])}")

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    label_id = label.item()
    
    class_name = id_to_class.get(label_id, f"Unknown({label_id})")
    color = colors[label_id % len(colors)]
    
    xmin, ymin, xmax, ymax = box
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    
    text = f"{class_name}: {score:.2f}"
    ax.text(xmin, ymin, text, fontsize=10, color='black',
            bbox=dict(facecolor=color, alpha=0.5, edgecolor='none'))

plt.axis('off')
plt.show()