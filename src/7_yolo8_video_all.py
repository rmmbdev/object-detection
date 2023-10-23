# dockerfile v1.02
import argparse
import glob
import os
import subprocess
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
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

warnings.filterwarnings("ignore")
ROOT = '/code/'
WEIGHTS_PATH = os.path.join(ROOT, 'models')


def learning_rate(args, params):
    def fn(x):
        return (1 - x / args.epochs) * (1.0 - params['lrf']) + params['lrf']

    return fn


def get_file_names(folder_names):
    result = []
    for folder_name in folder_names:
        result.extend([os.path.join(ROOT, folder_name, f) for f in os.listdir(folder_name)])

    return result


@torch.no_grad()
def test(args, params, total_frames_sample, clean, threshold, batch_size, model=None):
    folders = [
        os.path.join(ROOT, "data", 'VIS_Onboard', 'VIS_Onboard', 'Videos'),
        os.path.join(ROOT, "data", 'VIS_Onshore', 'VIS_Onshore', 'Videos'),
    ]
    filenames = get_file_names(folders)
    filenames = filenames[15:]
    dataset = Dataset(filenames, args.input_size, params, total_frames_sample)
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=Dataset.collate_fn
    )

    if model is None:
        model = torch.load(os.path.join(WEIGHTS_PATH, "v8_x.pt"), map_location='cuda')['model'].float()

    model.half()
    model.eval()

    file_index = 0
    sample_index = 0
    # buffers = []
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

                if conf > threshold:
                    sample_with_boxes = draw_rectangle_with_text_wrt_points(
                        sample_with_boxes,
                        int(box[0]),
                        int(box[1]),
                        int(box[2]),
                        int(box[3]),
                        params['names'][int(cls)],
                    )

            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(sample)
            axs[0].set_title('Original')

            axs[1].imshow(sample_with_boxes)
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

            # fig.savefig(
            #     buffer,
            #     bbox_inches='tight',
            #     dpi=300,
            #     format='png'
            # )
            # buffer.seek(0)
            # img = PIL.Image.open(buffer)
            # buffers.append(numpy.array(img))
            #
            # buffer.close()
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
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--total-frames-sample', default=-1, type=int)
    parser.add_argument('--clean', default=True, type=bool)
    parser.add_argument('--threshold', default=0.2, type=float)

    args = parser.parse_args()

    setup_seed()
    setup_multi_processes()

    with open('4_args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    test(args, params, args.total_frames_sample, args.clean, args.threshold, args.batch_size)

    print("Done!")


if __name__ == "__main__":
    main()
