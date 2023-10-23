# dockerfile v1.03
import argparse
import os
import time

import cv2
import numpy as np
import torch
import torchvision
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision.transforms import ToTensor

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
def convert_detections(detections, threshold, classes):
    # Get the bounding boxes, labels and scores from the detections dictionary.
    boxes = detections["boxes"].cpu().numpy()
    labels = detections["labels"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()
    lbl_mask = np.isin(labels, classes)
    scores = scores[lbl_mask]
    # Filter out low confidence scores and non-person classes.
    mask = scores > threshold
    boxes = boxes[lbl_mask][mask]
    scores = scores[mask]
    labels = labels[lbl_mask][mask]

    # Convert boxes to [x1, y1, w, h, score] format.
    final_boxes = []
    for i, box in enumerate(boxes):
        # Append ([x, y, w, h], score, label_string).
        final_boxes.append(
            (
                [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                scores[i],
                str(labels[i])
            )
        )

    return final_boxes


# Function for bounding box and ID annotation.
def annotate(tracks, frame, resized_frame, frame_width, frame_height, colors):
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        track_class = track.det_class
        x1, y1, x2, y2 = track.to_ltrb()
        p1 = (int(x1 / resized_frame.shape[1] * frame_width), int(y1 / resized_frame.shape[0] * frame_height))
        p2 = (int(x2 / resized_frame.shape[1] * frame_width), int(y2 / resized_frame.shape[0] * frame_height))
        # Annotate boxes.
        color = colors[int(track_class)]
        cv2.rectangle(
            frame,
            p1,
            p2,
            color=(int(color[0]), int(color[1]), int(color[2])),
            thickness=2
        )
        # Annotate ID.
        cv2.putText(
            frame, f"ID: {track_id}",
            (p1[0], p1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA
        )
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        default='input/mvmhat_1_1.mp4',
        help='path to input video',
    )
    parser.add_argument(
        '--imgsz',
        default=None,
        help='image resize, 640 will resize images to 640x640',
        type=int
    )
    parser.add_argument(
        '--model',
        default='fasterrcnn_resnet50_fpn_v2',
        help='model name',
        choices=[
            'fasterrcnn_resnet50_fpn_v2',
            'fasterrcnn_resnet50_fpn',
            'fasterrcnn_mobilenet_v3_large_fpn',
            'fasterrcnn_mobilenet_v3_large_320_fpn',
            'fcos_resnet50_fpn',
            'ssd300_vgg16',
            'ssdlite320_mobilenet_v3_large',
            'retinanet_resnet50_fpn',
            'retinanet_resnet50_fpn_v2'
        ]
    )
    parser.add_argument(
        '--threshold',
        default=0.8,
        help='score threshold to filter out detections',
        type=float
    )
    parser.add_argument(
        '--embedder',
        default='mobilenet',
        help='type of feature extractor to use',
        choices=[
            "mobilenet",
            "torchreid",
            "clip_RN50",
            "clip_RN101",
            "clip_RN50x4",
            "clip_RN50x16",
            "clip_ViT-B/32",
            "clip_ViT-B/16"
        ]
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='visualize results in real-time on screen'
    )
    parser.add_argument(
        '--cls',
        nargs='+',
        default=[9,],
        help='which classes to track',
        type=int
    )
    args = parser.parse_args()
    np.random.seed(42)
    OUT_DIR = 'outputs'
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    COLORS = np.random.randint(0, 255, size=(len(COCO_91_CLASSES), 3))
    print(f"Tracking: {[COCO_91_CLASSES[idx] for idx in args.cls]}")
    print(f"Detector: {args.model}")
    print(f"Re-ID embedder: {args.embedder}")
    # Load model.
    model = getattr(torchvision.models.detection, args.model)(weights='DEFAULT')
    # Set model to evaluation mode.
    model.eval().to(device)
    # Initialize a SORT tracker object.
    tracker = DeepSort(max_age=30, embedder=args.embedder)

    VIDEO_PATH = os.path.join(ROOT, "data", 'VIS_Onboard', 'VIS_Onboard', 'Videos', 'MVI_0788_VIS_OB.avi')
    # VIDEO_PATH = os.path.join(ROOT, "data", 'VIS_Onshore', 'VIS_Onshore', 'Videos', 'MVI_1582_VIS.avi')
    # VIDEO_PATH = os.path.join(ROOT, "data", "input", 'mvmhat_1_1.mp4')
    # VIDEO_PATH = args.input

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_fps = int(cap.get(5))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    save_name = VIDEO_PATH.split(os.path.sep)[-1].split('.')[0]
    # Define codec and create VideoWriter object.
    out = cv2.VideoWriter(
        f"{OUT_DIR}/{save_name}_{args.model}_{args.embedder}.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'), frame_fps,
        (frame_width, frame_height)
    )
    frame_count = 0  # To count total frames.
    total_fps = 0  # To get the final frames per second.
    while cap.isOpened():
        # Read a frame
        ret, frame = cap.read()
        if ret:
            if args.imgsz != None:
                resized_frame = cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    (args.imgsz, args.imgsz)
                )
            else:
                resized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert frame to tensor and send it to device (cpu or cuda).
            frame_tensor = ToTensor()(resized_frame).to(device)

            start_time = time.time()
            # Feed frame to model and get detections.
            det_start_time = time.time()
            with torch.no_grad():
                detections = model([frame_tensor])[0]
            det_end_time = time.time()

            det_fps = 1 / (det_end_time - det_start_time)

            # Convert detections to Deep SORT format.
            detections = convert_detections(detections, args.threshold, args.cls)

            # Update tracker with detections.
            track_start_time = time.time()
            tracks = tracker.update_tracks(detections, frame=frame)
            track_end_time = time.time()
            track_fps = 1 / (track_end_time - track_start_time)

            end_time = time.time()
            fps = 1 / (end_time - start_time)
            # Add `fps` to `total_fps`.
            total_fps += fps
            # Increment frame count.
            frame_count += 1

            print(f"Frame {frame_count}/{frames}",
                  f"Detection FPS: {det_fps:.1f},",
                  f"Tracking FPS: {track_fps:.1f}, Total FPS: {fps:.1f}")
            # Draw bounding boxes and labels on frame.
            if len(tracks) > 0:
                frame = annotate(
                    tracks,
                    frame,
                    resized_frame,
                    frame_width,
                    frame_height,
                    COLORS
                )
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (int(20), int(40)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            out.write(frame)
            if args.show:
                # Display or save output frame.
                cv2.imshow("Output", frame)
                # Press q to quit.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            break
    # Release resources.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
