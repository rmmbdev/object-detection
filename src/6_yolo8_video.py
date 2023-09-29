# dockerfile v1
import argparse
import os
import subprocess
import warnings

import cv2
import numpy
import torch
import yaml
from torch import Tensor
from torch.utils import data

from image_utils import draw_rectangle_with_text_wrt_points
from src.utils.dataset_video import Dataset
from src.utils.util import (
    non_max_suppression,
    setup_seed,
    setup_multi_processes
)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

warnings.filterwarnings("ignore")
ROOT = '/code/'
WEIGHTS_PATH = os.path.join(ROOT, 'models')


def learning_rate(args, params):
    def fn(x):
        return (1 - x / args.epochs) * (1.0 - params['lrf']) + params['lrf']

    return fn


@torch.no_grad()
def test(args, params, model=None):
    filenames = []
    filenames.append(os.path.join(ROOT, 'data', 'VIS_Onboard', 'VIS_Onboard', 'Videos', 'MVI_0790_VIS_OB.avi'))
    filenames.append(os.path.join(ROOT, 'data', 'VIS_Onboard', 'VIS_Onboard', 'Videos', 'MVI_0792_VIS_OB.avi'))
    dataset = Dataset(filenames, args.input_size, params, )
    loader = data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=Dataset.collate_fn
    )

    if model is None:
        model = torch.load(os.path.join(WEIGHTS_PATH, "v8_x.pt"), map_location='cuda')['model'].float()

    model.half()
    model.eval()

    plots = []
    file_index = 0
    for samples, targets, shapes in loader:
        samples = samples.cuda()
        targets = targets.cuda()
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255  # 0 - 255 to 0.0 - 1.0
        _, _, height, width = samples.shape  # batch size, channels, height, width

        # Inference
        outputs: Tensor = model(samples)

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height)).cuda()  # to pixels
        outputs = non_max_suppression(outputs, 0.001, 0.65)

        # Visualise
        samples = samples.cpu()
        for i, output in enumerate(outputs):
            sample = samples[i].T.numpy()
            sample *= 255
            sample = sample.astype(numpy.uint8)
            sample = sample.transpose((1, 0, 2))

            output = output.cpu()
            sample_with_boxes = sample.copy()
            for obj in output:
                box = obj[:4].numpy()
                conf = obj[4].numpy()
                cls = obj[5].numpy()

                if conf > 0.01:
                    sample_with_boxes = draw_rectangle_with_text_wrt_points(
                        sample_with_boxes,
                        int(box[0]),
                        int(box[1]),
                        int(box[2]),
                        int(box[3]),
                        str(cls),
                    )

            # fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            # axs[0].imshow(sample)
            # axs[0].set_title('Original')
            #
            # axs[1].imshow(sample_with_boxes)
            # axs[1].set_title('Detected')

            plt.subplot(1, 2, 1)
            plt.imshow(sample)
            plt.title('Original')

            plt.subplot(1, 2, 2)
            plt.imshow(sample_with_boxes)
            plt.title('Detected')
            #
            # plots.append(fig)
            plt.savefig(os.path.join(ROOT, "results", f"vid{file_index}-file{i}.png"))
            plots.append(plt)

            if len(plots) >= loader.dataset.video_frames[filenames[file_index]]:
                # save plots
                result_root = os.path.join(ROOT, "results")

                # subprocess.call([
                #     'local/bin/ffmpeg',
                #     '-framerate', '8',
                #     '-i', 'file%02d.png',
                #     '-r', '30',
                #     '-pix_fmt', 'yuv420p',
                #     os.path.join(result_root, 'video_name.mp4')
                # ])

                # def update(frame):
                #     plt.clf()  # Clear the current plot
                #     plt.imshow(plots[frame].canvas.buffer_rgba())
                #     plt.axis('off')  # Turn off axis
                #     plt.title(f'Frame {frame + 1}')
                #
                # fig = plt.figure()
                #
                # # Create an animation object
                # animation = FuncAnimation(fig, update, frames=len(plots), repeat=False)
                #
                # # Save the animation as an mp4 video
                # video_output_path = os.path.join(
                #     ROOT, "results", f"{filenames[file_index]}".replace(".avi", "-output.mp4")
                # )
                # animation.save(video_output_path, writer='ffmpeg')

                # video = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc('A', 'V', 'C', '1'), 1,
                #                         (mat.shape[0], mat.shape[1]))

                file_index += 1
                plots = []
            # plots.append(fig)

    return plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=128, type=int)

    args = parser.parse_args()

    setup_seed()
    setup_multi_processes()

    with open('4_args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    plots = test(args, params)

    print("Done!")


if __name__ == "__main__":
    main()
