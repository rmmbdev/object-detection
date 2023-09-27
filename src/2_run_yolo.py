import os

from ultralytics import YOLO
from ultralytics.engine.results import Results

ROOT = '/code/'


def main():
    model = YOLO(os.path.join(ROOT, 'models', "yolov8n.pt"))
    results = model(os.path.join(ROOT, 'data', 'bus.jpg'))
    details: Results = results[0]
    print("classes:", details.boxes.cls)
    print("location:", details.boxes.xywh)


if __name__ == '__main__':
    main()
