# dockerfile v1.04
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101

ROOT = '/code/'

# Load the pre-trained DeepLabV3 model
model = deeplabv3_resnet101(pretrained=True)
model.eval()

# Define a transform to preprocess the input image
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess an example image
image_path = os.path.join(ROOT, 'data', 'bus.jpg')
img = Image.open(image_path)
img = transform(img).unsqueeze(0)

# Perform inference
with torch.no_grad():
    out = model(img)

output = out['out'][0]
output_predictions = output.argmax(0).cpu().numpy()

coco_classes = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
num_classes = len(coco_classes)
cmap = plt.cm.get_cmap('viridis', num_classes)

# Visualize the segmentation mask
plt.imshow(output_predictions, cmap=cmap, alpha=0.7)

for class_idx in range(num_classes):  # Start from 1 to skip background
    mask = np.where(output_predictions == class_idx)
    center_y, center_x = np.mean(mask[0]), np.mean(mask[1])
    plt.text(center_x, center_y, coco_classes[class_idx], color='white',
             fontsize=8, ha='center', va='center', bbox=dict(facecolor=cmap(class_idx), alpha=0.7))

plt.savefig(os.path.join(ROOT, "results", f"bus_segmented.jpg"))
