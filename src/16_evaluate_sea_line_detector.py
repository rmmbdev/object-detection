# dockerfile v1.04
import argparse
import glob
import os
import subprocess
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
import yaml
from torch import Tensor
from torch.utils import data
from torchvision import models

from image_utils import draw_point
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
def evaluate(
    args,
    params,
    total_frames_sample,
    clean,
    threshold,
    batch_size,
    pos,
    model,
):
    filenames = [
        os.path.join(ROOT, "data", 'lab', 'processed', "vid-1-1.mp4"),
        os.path.join(ROOT, "data", 'lab', 'processed', "vid-1-8.mp4"),
        os.path.join(ROOT, "data", 'lab', 'processed', "vid-2-25.mp4"),
    ]

    dataset = Dataset(filenames, args.input_size, params, total_frames_sample, pos=pos)
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=Dataset.collate_fn
    )

    model.eval()

    file_index = 0
    sample_index = 0
    for samples, targets, shapes in loader:
        samples = samples.cuda()
        outputs: Tensor = model(samples)

        samples = samples.cpu()
        for i, output in enumerate(outputs):
            sample = samples[i].T.numpy()
            # sample *= 255
            # sample = sample.astype(numpy.uint8)
            sample = sample.transpose((1, 0, 2))

            mean = numpy.array([0.485, 0.456, 0.406])
            std = numpy.array([0.229, 0.224, 0.225])
            original_image = sample * std + mean
            original_image = (original_image * 255).astype(numpy.uint8)

            output = output.cpu()

            if pos == 0:
                sample_with_dot = draw_point(original_image.copy(), 0, int(output))
            elif pos == 2:
                sample_with_dot = draw_point(original_image.copy(), args.input_size, int(output))
            else:
                sample_with_dot = draw_point(original_image.copy(), args.input_size // 2, int(output))

            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(original_image)
            axs[0].set_title('Original')

            axs[1].imshow(sample_with_dot)
            axs[1].set_title('Detected')

            width_inches = 9.03  # Adjust as needed
            height_inches = 6.01  # Adjust as needed
            fig.set_size_inches(width_inches, height_inches)

            # buffer = BytesIO()
            fig.savefig(
                os.path.join(ROOT, "results", f"vid{file_index}-file{sample_index}.png"),
                bbox_inches='tight',
                dpi=300
            )
            plt.close(fig)

            sample_index += 1

            if sample_index >= loader.dataset.video_frames[filenames[file_index]]:
                images_path = os.path.join(ROOT, "results")
                images_pattern = f'{images_path}/vid{file_index}-file%d.png'

                video_output_path = os.path.join(
                    ROOT, "results", f"{Path(filenames[file_index]).name}".replace(".avi", "-output.mp4")
                )

                exit_code = subprocess.call([
                    'ffmpeg',
                    '-framerate', '8',
                    '-i', images_pattern,
                    '-r', '30',
                    '-pix_fmt', 'yuv420p',
                    video_output_path,
                    '-y'
                ])

                print("Video saved as", video_output_path, "with exit code:", exit_code)

                file_index += 1
                sample_index = 0

                if clean:
                    # remove image files
                    # images_path = os.path.join(ROOT, "results")
                    # images_pattern = f'*.png'
                    # matching_files = glob.glob(os.path.join(images_path, images_pattern))
                    images_pattern = images_pattern.replace('%d', '*')
                    matching_files = glob.glob(images_pattern)
                    print("Files:", matching_files)
                    for file_path in matching_files:
                        try:
                            os.remove(file_path)
                        except OSError as e:
                            print(f"Error: {file_path} - {e}")
                # Build video
                # clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(buffers, fps=10)
                #
                # video_output_path = os.path.join(
                #     ROOT, "results", f"{Path(filenames[file_index]).name}".replace(".avi", "-output.mp4")
                # )
                # # Write video
                # clip.write_videofile(video_output_path)
                # buffers = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--total-frames-sample', default=-1, type=int)
    parser.add_argument('--clean', default=True, type=bool)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--pos', default=2, type=int)
    parser.add_argument('--epochs', default=10, type=int)

    args = parser.parse_args()

    setup_seed()
    setup_multi_processes()

    with open('15_args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    model = torch.load(os.path.join(ROOT, "models", f"model-20231224214634-4.pth"))

    evaluate(
        args=args,
        params=params,
        total_frames_sample=args.total_frames_sample,
        clean=args.clean,
        threshold=args.threshold,
        batch_size=args.batch_size,
        pos=args.pos,
        model=model,
    )

    print("Done!")


if __name__ == "__main__":
    main()
