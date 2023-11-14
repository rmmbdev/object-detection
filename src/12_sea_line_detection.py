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
from src.utils.two_objectives_horizon_detection import extract_horizon, get_plane_indicator_coord

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
        default=640,
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

    global_img_reduction = 0.1
    global_angles = (-90, 91, 5)
    # global_distances = (5, 100, 5)
    global_distances = (10, 100, 10)
    global_buffer_size = 3

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_fps = int(cap.get(5))
    # frame_fps = 2
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    save_name = VIDEO_PATH.split(os.path.sep)[-1].split('.')[0]
    # Define codec and create VideoWriter object.
    out = cv2.VideoWriter(
        f"{OUT_DIR}/{save_name}_line.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'), frame_fps,
        (args.imgsz, args.imgsz)
    )
    frame_count = 0  # To count total frames.
    total_fps = 0  # To get the final frames per second.
    while cap.isOpened():
        if frame_count > 10:
            break
        # Read a frame
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if args.imgsz != None:
                frame = cv2.resize(frame, (args.imgsz, args.imgsz))

            start_time = time.time()
            # Feed frame to model and get detections.
            det_start_time = time.time()
            global_search = extract_horizon(
                frame,
                angles=global_angles,
                distances=global_distances,
                buffer_size=global_buffer_size,
                local_objective=0
            )
            # GLOBAL OBJECTIVE OPTIMIZATION SURFACE
            objective_1 = np.max((global_search[:, :, 0] - global_search[:, :, 1]), 0) / (global_search[:, :, 2])
            X = np.arange(0, len(range(*global_angles)), 1)
            Y = np.arange(0, len(range(*global_distances)), 1)
            X, Y = np.meshgrid(Y, X)
            Z = objective_1

            # LOCAL OBJECTIVE SETTINGS
            above_two_sigma = (2 * np.nanstd(objective_1)) + np.nanmean(objective_1)

            local_angles = global_search[np.where(objective_1 > above_two_sigma)[0],
            np.where(objective_1 > above_two_sigma)[1]][:, 6]
            local_distances = global_search[np.where(objective_1 > above_two_sigma)[0],
            np.where(objective_1 > above_two_sigma)[1]][:, 7]
            local_angle_range = (int(np.min(local_angles)) - 2,
                                 int(np.max(local_angles)) + 3, 1)
            local_distance_range = (int(np.min(local_distances)) - 2,
                                    int(np.max(local_distances)) + 3, 1)

            local_img_reduction = 0.25
            local_angles = local_angle_range
            local_distances = local_distance_range
            local_buffer_size = 5

            # LOCAL OBJECTIVE MAIN ROUTINE
            print("Evaluating", (len(range(*local_angle_range)) * len(range(*local_distance_range))), "candidates...")

            local_search = extract_horizon(
                frame,
                angles=local_angles,
                distances=local_distances,
                buffer_size=local_buffer_size,
                local_objective=1
            )  # 2m5s

            # LOCAL OBJECTIVE OPTIMIZATION SURFACE
            objective_2 = (local_search[:, :, 4] - local_search[:, :, 5]) ** 2 / local_search[:, :, 2]

            horizon_line = local_search[np.unravel_index(objective_2.argmax(), objective_2.shape)]

            # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            line_coordinates = get_plane_indicator_coord(
                frame,
                int(horizon_line[6]),
                horizon_line[7] / 100,
                0
            )[2:4]
            frame_annotated = cv2.line(
                frame,
                (line_coordinates[0][0], line_coordinates[0][1]),
                (line_coordinates[1][0], line_coordinates[1][1]),
                (0, 0, 255),
                2
            )

            frame = cv2.cvtColor(frame_annotated, cv2.COLOR_GRAY2BGR)
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
