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

from src.utils.util import detect_horizon_line, draw_horizon

ROOT = '/code/'


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
        '--show',
        action='store_true',
        help='visualize results in real-time on screen'
    )
    parser.add_argument(
        '--cls',
        nargs='+',
        default=[9, ],
        help='which classes to track',
        type=int
    )
    args = parser.parse_args()
    np.random.seed(42)
    OUT_DIR = 'outputs'
    os.makedirs(OUT_DIR, exist_ok=True)

    VIDEO_PATH = os.path.join(ROOT, "data", 'VIS_Onboard', 'VIS_Onboard', 'Videos', 'MVI_0788_VIS_OB.avi')
    # VIDEO_PATH = os.path.join(ROOT, "data", 'VIS_Onshore', 'VIS_Onshore', 'Videos', 'MVI_1622_VIS.avi')
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
        f"{OUT_DIR}/{save_name}_line.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'), frame_fps,
        (frame_width, frame_height)
    )
    frame_count = 0  # To count total frames.
    total_fps = 0  # To get the final frames per second.
    while cap.isOpened():
        # Read a frame
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if args.imgsz != None:
                frame = cv2.resize(frame, (args.imgsz, args.imgsz))

            start_time = time.time()
            # Feed frame to model and get detections.
            det_start_time = time.time()
            frame = draw_horizon(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            det_end_time = time.time()

            det_fps = 1 / (det_end_time - det_start_time)

            end_time = time.time()
            fps = 1 / (end_time - start_time)
            # Add `fps` to `total_fps`.
            total_fps += fps
            # Increment frame count.
            frame_count += 1

            print(
                f"Frame {frame_count}/{frames}",
                f"Detection FPS: {det_fps:.1f},",
            )
            # Draw bounding boxes and labels on frame.
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
