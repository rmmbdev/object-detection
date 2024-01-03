# dockerfile v1.04
import argparse
import json
import os
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch import Tensor
from torch.utils import data
from torchvision import models
import cv2

from src.image_utils import draw_point, save_image
from src.utils.dataset_image_train import Dataset
from src.utils.util import (
    setup_seed,
    setup_multi_processes
)
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = '/code/'
WEIGHTS_PATH = os.path.join(ROOT, 'models')


class ResNet50Model(nn.Module):
    def __init__(self):
        super().__init__()

        # ResNet50 backbone
        resnet50 = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet50.children())[:-2])
        for param in self.features.parameters():
            param.requires_grad = False
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def make_features_trainable(self, tail_layers_count=1):
        for params in self.features[-tail_layers_count:].parameters():
            params.requires_grad = True


class Resnet18BasedModel(nn.Module):
    def __init__(self):
        super(Resnet18BasedModel, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet18.children())[:-2])  # Exclude the last two layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 1)  # Regression output, 1 output node
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def learning_rate(args, params):
    def fn(x):
        return (1 - x / args.epochs) * (1.0 - params['lrf']) + params['lrf']

    return fn


def get_file_names(folder_names):
    result = []
    for folder_name in folder_names:
        result.extend([os.path.join(ROOT, folder_name, f) for f in os.listdir(folder_name) if ".jpeg" in f])

    return result


@torch.no_grad()
def predict(
    args,
    params,
    batch_size,
    pos,
    model,
):
    filenames = [
        os.path.join(ROOT, "data", 'lab', 'images', "vid-1-1-2.jpeg"),
        os.path.join(ROOT, "data", 'lab', 'test', "img-1.jpg"),
        os.path.join(ROOT, "data", 'lab', 'images', "vid-2-4-898.jpeg"),
    ]
    dataset = Dataset(filenames, args.input_size, params, pos=pos)
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=Dataset.collate_fn,
    )

    model.eval()

    index = 0
    for samples, targets, shapes in loader:
        samples = samples.cuda()
        outputs: Tensor = model(samples)

        for sample, output in zip(samples, outputs):
            pred = int(output.item())
            image = sample.cpu().numpy()
            image = image.transpose((1, 2, 0))

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            original_image = image * std + mean
            original_image = (original_image * 255).astype(np.uint8)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            if pos == 0:
                image_point = draw_point(original_image.copy(), 0, pred)
            elif pos == 2:
                image_point = draw_point(original_image.copy(), original_image.shape[1], pred)
            else:
                image_point = draw_point(original_image.copy(), original_image.shape[1] // 2, pred)

            file = filenames[index].split(os.path.sep)[-1]
            save_image(image_point, os.path.join(ROOT, "results", f"result-{file}"))
            index += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--total-frames-sample', default=-1, type=int)
    parser.add_argument('--clean', default=True, type=bool)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--pos', default=0, type=int)
    parser.add_argument('--epochs', default=5, type=int)

    args = parser.parse_args()

    setup_seed()
    setup_multi_processes()

    with open('15_args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    # define model here
    model = torch.load(os.path.join(ROOT, "models", f"model-20240103175733-59.pth"))

    predict(
        args=args,
        params=params,
        batch_size=args.batch_size,
        pos=args.pos,
        model=model,
    )

    print("Done!")


if __name__ == "__main__":
    main()
