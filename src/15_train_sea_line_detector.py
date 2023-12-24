# dockerfile v1.04
import argparse
import json
import os
import warnings
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch import Tensor
from torch.utils import data
from torchvision import models

from src.utils.dataset_video_train import Dataset
from src.utils.util import (
    setup_seed,
    setup_multi_processes
)

warnings.filterwarnings("ignore")
ROOT = '/code/'
WEIGHTS_PATH = os.path.join(ROOT, 'models')


class VGG19BasedModel(nn.Module):
    def __init__(self):
        super(VGG19BasedModel, self).__init__()
        self.features = models.vgg19(pretrained=True).features
        self.avgpool = models.vgg19(pretrained=True).avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1)  # Regression output, 1 output node
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Resnet18BasedModel(nn.Module):
    def __init__(self):
        super(Resnet18BasedModel, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet18.children())[:-2])  # Exclude the last two layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1)  # Regression output, 1 output node
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class EfficientNetV2SBasedModel(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNetV2SBasedModel, self).__init__()

        # Load EfficientNetV2-S model
        efficientnet_v2_s = models.efficientnet_v2_s(pretrained=True)
        self.features = efficientnet_v2_s.features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.num_features = 1280

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)  # Regression output, 1 output node
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def learning_rate(args, params):
    def fn(x):
        return (1 - x / args.epochs) * (1.0 - params['lrf']) + params['lrf']

    return fn


def get_file_names(folder_names):
    result = []
    for folder_name in folder_names:
        result.extend([os.path.join(ROOT, folder_name, f) for f in os.listdir(folder_name) if ".mp4" in f])

    return result


@torch.no_grad()
def train(
    args,
    params,
    total_frames_sample,
    clean,
    threshold,
    batch_size,
    pos,
    model,
    optimizer,
    criterion,
    lr_scheduler,
    epochs,
):
    experiment_date_time = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
    print(experiment_date_time)

    folders = [
        os.path.join(ROOT, "data", 'lab', 'processed'),
    ]
    filenames = get_file_names(folders)

    with open(os.path.join(ROOT, "data", 'lab', 'processed', 'labels.json'), mode='r') as fp:
        labels_raw = json.load(fp)

    filenames_filtered = []
    labels = {}
    old_pos = pos
    for idx, lbl_dict in enumerate(labels_raw):
        pos = old_pos
        if (idx == 0) and (pos == 1):
            pos = 2
        elif (idx == 0) and (pos == 2):
            pos = 1

        file_name = "-".join(lbl_dict['file_upload'].split('-')[1:])
        for fn in filenames:
            if file_name in fn:
                filenames_filtered.append(fn)
                labels[fn] = {
                    "frames_count": lbl_dict['annotations'][0]['result'][pos]['value']['framesCount'],
                    "value": lbl_dict['annotations'][0]['result'][pos]['value']['sequence']
                }
                break

    dataset = Dataset(filenames_filtered, args.input_size, params, total_frames_sample, labels, pos=pos)
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=Dataset.collate_fn
    )

    model.train()
    for epoch in range(epochs):
        file_index = 0
        sample_index = 0
        # buffers = []
        for samples, targets, shapes in loader:
            samples = samples.cuda()
            targets = targets.cuda()

            with torch.set_grad_enabled(True):
                outputs: Tensor = model(samples)
                loss = criterion(outputs, targets.float().view(-1, 1))

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Print progress
            print(f'Epoch:{epoch} -> loss:{loss.item()}')

        lr_scheduler.step()
        torch.save(model, os.path.join(ROOT, "models", f"model-{experiment_date_time}-{epoch}.pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--total-frames-sample', default=-1, type=int)
    parser.add_argument('--clean', default=True, type=bool)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--pos', default=2, type=int)
    parser.add_argument('--epochs', default=30, type=int)

    args = parser.parse_args()

    setup_seed()
    setup_multi_processes()

    with open('15_args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    # define model here
    model = Resnet18BasedModel().cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    criterion = nn.MSELoss()

    train(
        args=args,
        params=params,
        total_frames_sample=args.total_frames_sample,
        clean=args.clean,
        threshold=args.threshold,
        batch_size=args.batch_size,
        pos=args.pos,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        epochs=args.epochs,
    )

    print("Done!")


if __name__ == "__main__":
    main()
