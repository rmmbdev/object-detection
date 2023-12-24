import os
import random

import cv2
import math
import moviepy.editor as mpy
import numpy
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from src.image_utils import save_image, draw_point

FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'


class Dataset(data.Dataset):
    def __init__(self, filenames, input_size, params, total_frames_sample, labels=None, pos=0):
        self.params = params
        self.input_size = input_size
        self.filenames = filenames
        self.video_frames = {}
        self.video_sizes = {}
        self.pos = pos

        if total_frames_sample > 0:
            self.total_frames_sample = total_frames_sample
        else:
            self.total_frames_sample = np.Inf

        self.current_reader = None
        self.current_file_index = 0
        self.current_frame_index = 0

        self.load_files()
        self.labels = None
        if labels is not None:
            self.labels = self.generate_labels(labels)

    def load_files(self):
        print("Preloading dataset")
        total = len(self.filenames)
        for idx, fn in enumerate(self.filenames):
            print(f"{idx + 1}/{total} Loading file: [{fn}]")
            video_data = mpy.VideoFileClip(filename=fn, fps_source="fps")
            frame_count = int(video_data.fps * video_data.duration)

            self.video_frames[fn] = min(frame_count, self.total_frames_sample)
            self.video_sizes[fn] = video_data.size
            print(f"Frames: {self.video_frames[fn]}")

        print("Dataset total frames:", sum([v for k, v in self.video_frames.items()]))

    def generate_labels(self, labels):
        result = {}

        for k, v in self.video_frames.items():
            frames_labels = []
            for frame_idx in range(v):
                mapped_frame_index = int((frame_idx * labels[k]['frames_count']) / v)
                file_labels = labels[k]['value'] + [{'frame': 1000000}]
                height = self.video_sizes[k][1]
                for idx in range(len(file_labels)):
                    if file_labels[idx]['frame'] - 1 <= mapped_frame_index <= file_labels[idx + 1]['frame'] - 1:
                        # mapped_height = int((int(file_labels[idx]['y']) * self.input_size) / height)
                        mapped_height = int((int(file_labels[idx]['y']) * self.input_size) / 100)
                        frames_labels.append(mapped_height)
                        break

            result[k] = frames_labels

        return result

    def __getitem__(self, index):
        if self.current_reader is None:
            file_name = self.filenames[self.current_file_index]
            video = mpy.VideoFileClip(file_name)
            self.current_reader = video.iter_frames()
            self.current_frame_index = 0

        if self.current_frame_index >= self.video_frames[self.filenames[self.current_file_index]]:
            self.current_file_index += 1
            file_name = self.filenames[self.current_file_index]
            video = mpy.VideoFileClip(file_name)
            self.current_reader = video.iter_frames()
            self.current_frame_index = 0

        if index % 100 == 0:
            print("File:", self.current_file_index, " Frame", index)
        # print("File:", self.current_file_index, " Frame", index)

        image, shape = self.load_image(index)

        h, w = image.shape[:2]

        # Resize
        # image, ratio, pad = resize(image, self.input_size, augment=False)
        image = crop_resize(image, self.input_size, pos=self.pos)
        pad = (0, 0)

        shapes = shape, ((h / shape[0], w / shape[1]), pad)  # for COCO mAP rescaling

        target_value = 0
        if self.labels is not None:
            file_name = self.filenames[self.current_file_index]
            target_value = self.labels[file_name][self.current_frame_index]
            # target_value += pad[1]
        target = torch.from_numpy(np.array([target_value]))
        target = target.to(torch.float32)

        # if self.pos == 0:
        #     image = draw_point(image.copy(), 0, int(target_value))
        # elif self.pos == 2:
        #     image = draw_point(image.copy(), self.input_size, int(target_value))
        # else:
        #     image = draw_point(image.copy(), self.input_size // 2, int(target_value))
        # save_image(image)

        # Normalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_normalized = (image / 255.0 - mean) / std

        # Convert HWC to CHW, BGR to RGB
        sample = img_normalized.transpose((2, 0, 1))
        sample = numpy.ascontiguousarray(sample)
        normalized_image_tensor = torch.from_numpy(sample)

        normalized_image_tensor = normalized_image_tensor.to(torch.float32)

        self.current_frame_index += 1
        return normalized_image_tensor, target, shapes

    def __len__(self):
        total_frames = 0
        for _, v in self.video_frames.items():
            total_frames += v
        return total_frames

    def load_image(self, i):
        image = next(self.current_reader)
        h, w = image.shape[:2]
        r = self.input_size / max(h, w)
        if r != 1:
            image = cv2.resize(
                image,
                dsize=(int(w * r), int(h * r)),
                interpolation=cv2.INTER_LINEAR
            )
        return image, (h, w)

    @staticmethod
    def collate_fn(batch):
        samples, targets, shapes = zip(*batch)
        # for i, item in enumerate(targets):
        #     item[:, 0] = i  # add target image index
        return torch.stack(samples, 0), torch.cat(targets, 0), shapes

    @staticmethod
    def load_label(filenames):
        path = f'{os.path.dirname(filenames[0])}.cache'
        if os.path.exists(path):
            return torch.load(path)
        x = {}
        for filename in filenames:
            try:
                # verify images
                with open(filename, 'rb') as f:
                    image = Image.open(f)
                    image.verify()  # PIL verify
                shape = image.size  # image size
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert image.format.lower() in FORMATS, f'invalid image format {image.format}'

                # verify labels
                a = f'{os.sep}images{os.sep}'
                b = f'{os.sep}labels{os.sep}'
                if os.path.isfile(b.join(filename.rsplit(a, 1)).rsplit('.', 1)[0] + '.txt'):
                    with open(b.join(filename.rsplit(a, 1)).rsplit('.', 1)[0] + '.txt') as f:
                        label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        label = numpy.array(label, dtype=numpy.float32)
                    nl = len(label)
                    if nl:
                        assert label.shape[1] == 5, 'labels require 5 columns'
                        assert (label >= 0).all(), 'negative label values'
                        assert (label[:, 1:] <= 1).all(), 'non-normalized coordinates'
                        _, i = numpy.unique(label, axis=0, return_index=True)
                        if len(i) < nl:  # duplicate row check
                            label = label[i]  # remove duplicates
                    else:
                        label = numpy.zeros((0, 5), dtype=numpy.float32)
                else:
                    label = numpy.zeros((0, 5), dtype=numpy.float32)
                if filename:
                    x[filename] = [label, shape]
            except FileNotFoundError:
                pass
        torch.save(x, path)
        return x


def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    # Convert nx4 boxes
    # from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y


def xy2wh(x, w=640, h=640):
    # warning: inplace clip
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1E-3)  # x1, x2
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1E-3)  # y1, y2

    # Convert nx4 boxes
    # from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    y = numpy.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def resample():
    choices = (cv2.INTER_AREA,
               cv2.INTER_CUBIC,
               cv2.INTER_LINEAR,
               cv2.INTER_NEAREST,
               cv2.INTER_LANCZOS4)
    return random.choice(seq=choices)


def augment_hsv(image, params):
    # HSV color-space augmentation
    h = params['hsv_h']
    s = params['hsv_s']
    v = params['hsv_v']

    r = numpy.random.uniform(-1, 1, 3) * [h, s, v] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    x = numpy.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype('uint8')
    lut_s = numpy.clip(x * r[1], 0, 255).astype('uint8')
    lut_v = numpy.clip(x * r[2], 0, 255).astype('uint8')

    im_hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed


def crop_resize(image, input_size, pos=0):
    height, width, _ = image.shape

    # Calculate the side length of the square
    side_length = min(height, width)

    if pos == 0:
        # Crop a square from the left
        cropped = image[:, :side_length, :]
    elif pos == 1:
        # Crop a square from the center
        start_x = (width - side_length) // 2
        start_y = (height - side_length) // 2
        cropped = image[start_y:start_y + side_length, start_x:start_x + side_length, :]
    else:
        # Crop a square from the right
        cropped = image[:, -side_length:, :]

    resized_image = cv2.resize(cropped, (input_size, input_size))
    return resized_image


def resize(image, input_size, augment):
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(input_size / shape[0], input_size / shape[1])
    if not augment:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2

    if shape[::-1] != pad:  # resize
        image = cv2.resize(
            image,
            dsize=pad,
            interpolation=resample() if augment else cv2.INTER_LINEAR
        )
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border
    return image, (r, r), (w, h)


def candidates(box1, box2):
    # box1(4,n), box2(4,n)
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    aspect_ratio = numpy.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.1) & (aspect_ratio < 100)


def random_perspective(samples, targets, params, border=(0, 0)):
    h = samples.shape[0] + border[0] * 2
    w = samples.shape[1] + border[1] * 2

    # Center
    center = numpy.eye(3)
    center[0, 2] = -samples.shape[1] / 2  # x translation (pixels)
    center[1, 2] = -samples.shape[0] / 2  # y translation (pixels)

    # Perspective
    perspective = numpy.eye(3)

    # Rotation and Scale
    rotate = numpy.eye(3)
    a = random.uniform(-params['degrees'], params['degrees'])
    s = random.uniform(1 - params['scale'], 1 + params['scale'])
    rotate[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    shear = numpy.eye(3)
    shear[0, 1] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)
    shear[1, 0] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)

    # Translation
    translate = numpy.eye(3)
    translate[0, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * w
    translate[1, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * h

    # Combined rotation matrix, order of operations (right to left) is IMPORTANT
    matrix = translate @ shear @ rotate @ perspective @ center
    if (border[0] != 0) or (border[1] != 0) or (matrix != numpy.eye(3)).any():  # image changed
        samples = cv2.warpAffine(samples, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))

    # Transform label coordinates
    n = len(targets)
    if n:
        xy = numpy.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ matrix.T  # transform
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = numpy.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, w)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, h)

        # filter candidates
        indices = candidates(box1=targets[:, 1:5].T * s, box2=new.T)
        targets = targets[indices]
        targets[:, 1:5] = new[indices]

    return samples, targets


def mix_up(image1, label1, image2, label2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    alpha = numpy.random.beta(32.0, 32.0)  # mix-up ratio, alpha=beta=32.0
    image = (image1 * alpha + image2 * (1 - alpha)).astype(numpy.uint8)
    label = numpy.concatenate((label1, label2), 0)
    return image, label


class Albumentations:
    def __init__(self):
        self.transform = None
        try:
            import albumentations as album

            transforms = [album.Blur(p=0.01),
                          album.CLAHE(p=0.01),
                          album.ToGray(p=0.01),
                          album.MedianBlur(p=0.01)]
            self.transform = album.Compose(transforms,
                                           album.BboxParams('yolo', ['class_labels']))

        except ImportError:  # package not installed, skip
            pass

    def __call__(self, image, label):
        if self.transform:
            x = self.transform(image=image,
                               bboxes=label[:, 1:],
                               class_labels=label[:, 0])
            image = x['image']
            label = numpy.array([[c, *b] for c, b in zip(x['class_labels'], x['bboxes'])])
        return image, label
