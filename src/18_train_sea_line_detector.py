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

from src.utils.dataset_image_train import Dataset
from src.utils.util import (
    setup_seed,
    setup_multi_processes
)
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = '/code/'
WEIGHTS_PATH = os.path.join(ROOT, 'models')


class VGG19BasedModel(nn.Module):
    def __init__(self):
        super(VGG19BasedModel, self).__init__()
        self.features = models.vgg19(pretrained=True).features
        for params in self.features.parameters():
            params.requires_grad = False
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
        for params in self.features.parameters():
            params.requires_grad = False
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
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


class EfficientNetB0BasedModel(nn.Module):
    def __init__(self):
        super(EfficientNetB0BasedModel, self).__init__()
        model = models.efficientnet_b0(pretrained=True)
        for params in model.parameters():
            params.requires_grad = False
        model.classifier[1] = nn.Linear(in_features=1280, out_features=1)
        self.model = model

    def forward(self, x):
        x = self.model(x)
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


def train_epoch(model, loader, criterion, optimizer, lr_scheduler):
    losses = []
    predicts = []
    labels = []
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
        losses.append(loss.item())
        if len(losses) % 10 == 0:
            print("loss", loss.item())

        targets = targets.cpu()
        labels.extend(targets)

        outputs = outputs.cpu()
        predicts.extend(outputs)

    lr_scheduler.step()

    return losses, predicts, labels, model, loader, criterion, optimizer, lr_scheduler


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
        os.path.join(ROOT, "data", 'lab', 'images'),
    ]
    filenames = get_file_names(folders)

    labels = pd.read_csv(os.path.join(ROOT, "data", 'lab', 'images', "labels.csv"))
    dataset = Dataset(filenames, args.input_size, params, labels=labels, pos=pos)
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=Dataset.collate_fn,
    )

    model.train()
    epoch_number = 0

    print("####################### Only FC ###########################")
    for epoch in range(3):
        losses, predicts, labels, model, loader, criterion, optimizer, lr_scheduler = train_epoch(
            model,
            loader,
            criterion,
            optimizer,
            lr_scheduler
        )
        print(f'*************** Epoch:{epoch} -> mean loss:{np.mean(losses)}')
        torch.save(model, os.path.join(ROOT, "models", f"model-{experiment_date_time}-{epoch_number}.pth"))
        print("Write labels and predicts")
        with open(os.path.join(ROOT, "results", f"diff-{epoch_number}.txt"), mode='w') as f:
            for lbl, pred in zip(labels, predicts):
                f.write(f"{lbl.item()}-{pred.item()}\n")
        epoch_number += 1

    print("####################### With FEATURES ###########################")
    tails = 20
    for t in range(1, tails):
        print(f"*********************** With FEATURES {t} ***********************")
        model.make_features_trainable(tail_layers_count=t * 2)
        for epoch in range(3):
            losses, predicts, labels, model, loader, criterion, optimizer, lr_scheduler = train_epoch(
                model,
                loader,
                criterion,
                optimizer,
                lr_scheduler
            )
            print(f'*************** Epoch:{epoch} -> mean loss:{np.mean(losses)}')
            torch.save(model, os.path.join(ROOT, "models", f"model-{experiment_date_time}-{epoch_number}.pth"))
            print("Write labels and predicts")
            with open(os.path.join(ROOT, "results", f"diff-{epoch_number}.txt"), mode='w') as f:
                for lbl, pred in zip(labels, predicts):
                    f.write(f"{lbl.item()}-{pred.item()}\n")
            epoch_number += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=256, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--total-frames-sample', default=-1, type=int)
    parser.add_argument('--clean', default=True, type=bool)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--pos', default=0, type=int)
    parser.add_argument('--epochs', default=2, type=int)

    args = parser.parse_args()

    setup_seed()
    setup_multi_processes()

    with open('15_args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    # define model here
    model = Resnet18BasedModel().cuda()
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    criterion = nn.HuberLoss()

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
