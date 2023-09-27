import os

from ultralytics import YOLO
from ultralytics.engine.results import Results
import matplotlib.pyplot as plt
import numpy as np

from image_utils import draw_rectangle_with_text_wrt_hw

ROOT = '/code/'


def main():
    model = YOLO(os.path.join(ROOT, 'models', "yolov8n.pt"))
    sample_path = os.path.join(ROOT, 'data', 'bus.jpg')
    results = model(sample_path)
    details: Results = results[0]

    image_numpy = details.orig_img

    object_index = 2
    image_numpy = draw_rectangle_with_text_wrt_hw(
        image_numpy,
        int(details.boxes.xywh[object_index][0]),
        int(details.boxes.xywh[object_index][1]),
        int(details.boxes.xywh[object_index][2]),
        int(details.boxes.xywh[object_index][3]),
        details.names[int(details.boxes.cls[object_index])]
    )

    image_numpy = image_numpy[:, :, [1, 2, 0]]
    plt.imshow(image_numpy)

    idx = 1
    plt.savefig(os.path.join(ROOT, "results", f"bus_{idx}.jpg"))


if __name__ == '__main__':
    main()
