# dockerfile v1.03
import argparse
import os
import time
import json
import pandas as pd

import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from torch import Tensor
from torchvision.transforms import ToTensor

from src.utils.util import (
    non_max_suppression
)
from src.utils.dataset_video import resize

COCO_91_CLASSES = [
    '__background__',
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

ROOT = '/code/'


# Define a function to convert detections to SORT format.


def main():
    filenames = [
        os.path.join(ROOT, "data", 'lab', 'processed', 'vid-1-1.mp4'),
        os.path.join(ROOT, "data", 'lab', 'processed', 'vid-1-2.mp4'),
        os.path.join(ROOT, "data", 'lab', 'processed', 'vid-1-3.mp4'),
        os.path.join(ROOT, "data", 'lab', 'processed', 'vid-1-4.mp4'),
        os.path.join(ROOT, "data", 'lab', 'processed', 'vid-1-5.mp4'),
        os.path.join(ROOT, "data", 'lab', 'processed', 'vid-1-6.mp4'),
        os.path.join(ROOT, "data", 'lab', 'processed', 'vid-1-7.mp4'),
        os.path.join(ROOT, "data", 'lab', 'processed', 'vid-1-8.mp4'),
        os.path.join(ROOT, "data", 'lab', 'processed', 'vid-1-9.mp4'),
        os.path.join(ROOT, "data", 'lab', 'processed', 'vid-1-10.mp4'),
        os.path.join(ROOT, "data", 'lab', 'processed', 'vid-1-11.mp4'),
        os.path.join(ROOT, "data", 'lab', 'processed', 'vid-2-1.mp4'),
        os.path.join(ROOT, "data", 'lab', 'processed', 'vid-2-2.mp4'),
        os.path.join(ROOT, "data", 'lab', 'processed', 'vid-2-4.mp4'),
    ]
    image_size = 640
    pad = False
    image_root = os.path.join(ROOT, "data", 'lab', 'images')

    frames_count = {}
    for video_path in filenames:
        video_name = video_path.split(os.path.sep)[-1].split('.')[0]
        cap = cv2.VideoCapture(video_path)
        frame_index = 0
        print("vide:", video_name)
        while cap.isOpened():
            # Read a frame
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            print("Frame:", frame_index)

            resized_frame = frame.copy()
            h, w = frame.shape[:2]
            r = image_size / max(h, w)
            if r != 1:
                resized_frame = cv2.resize(
                    frame,
                    dsize=(int(w * r), int(h * r)),
                    interpolation=cv2.INTER_LINEAR
                )

            if pad:
                resized_frame, ratio, pad = resize(resized_frame, image_size, augment=False)

            file_name = os.path.join(image_root, f"{video_name}-{frame_index}.jpeg")
            cv2.imwrite(file_name, resized_frame)

        # Release resources.
        cap.release()
        cv2.destroyAllWindows()
        frames_count[video_name] = frame_index

    with open(os.path.join(ROOT, "data", 'lab', 'processed', 'labels.json'), mode='r') as fp:
        labels_raw = json.load(fp)

    labels = {}
    for idx, lbl_dict in enumerate(labels_raw):
        file_name = "-".join(lbl_dict['file_upload'].split('-')[1:]).split('.')[0]
        for fn in filenames:
            if file_name in fn:
                labels[file_name] = {
                    "frames_count": lbl_dict['annotations'][0]['result'][0]['value']['framesCount'],
                    "left": lbl_dict['annotations'][0]['result'][0]['value']['sequence'],
                    "center": lbl_dict['annotations'][0]['result'][1]['value']['sequence'],
                    "right": lbl_dict['annotations'][0]['result'][2]['value']['sequence'],
                }
                break

    labels_rows = []
    for k, v in frames_count.items():
        frames_labels = []
        for frame_idx in range(v):
            mapped_frame_index = int((frame_idx * labels[k]['frames_count']) / v)
            labels[k]['left'] += [{'frame': 1000000}]
            labels[k]['center'] += [{'frame': 1000000}]
            labels[k]['right'] += [{'frame': 1000000}]
            label_row = {"file": f"{k}-{frame_idx + 1}.jpeg"}
            for idx in range(len(labels[k]['left'])):
                if (
                    labels[k]['left'][idx]['frame'] - 1
                    <= mapped_frame_index <=
                    labels[k]['left'][idx + 1]['frame'] - 1
                ):
                    mapped_height = labels[k]['left'][idx]['y']
                    label_row["left"] = mapped_height
                    break

            for idx in range(len(labels[k]['center'])):
                if (
                    labels[k]['center'][idx]['frame'] - 1 <=
                    mapped_frame_index <=
                    labels[k]['center'][idx + 1]['frame'] - 1
                ):
                    mapped_height = labels[k]['center'][idx]['y']
                    label_row["center"] = mapped_height
                    break

            for idx in range(len(labels[k]['right'])):
                if (
                    labels[k]['right'][idx]['frame'] - 1
                    <= mapped_frame_index <=
                    labels[k]['right'][idx + 1]['frame'] - 1
                ):
                    mapped_height = labels[k]['right'][idx]['y']
                    label_row["right"] = mapped_height
                    break

            labels_rows.append(label_row)

    labels_df: pd.DataFrame = pd.DataFrame(labels_rows)
    labels_df = labels_df.set_index("file")
    labels_df.to_csv(os.path.join(ROOT, "data", 'lab', 'images', 'labels.csv'))


if __name__ == '__main__':
    main()
