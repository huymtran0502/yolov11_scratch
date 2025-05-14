import math
import os
import random
from glob import glob
import cv2
import numpy as np
import torch
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset
from tqdm import tqdm

FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff'

class Yolo_Dataset(Dataset):
    def __init__(self, filenames, input_size, params=None, augment=False):
        self.params = params if params else {}
        self.augment = augment
        self.input_size = input_size

        # Read labels
        labels = self.load_label(filenames)
        self.labels = list(labels.values())
        self.filenames = list(labels.keys())  # update
        self.n = len(self.filenames)  # number of samples
        self.indices = range(self.n)
        self.albumentations = Albumentations()

    def __getitem__(self, index):
        index = self.indices[index]

        image, shape = self.load_image(index)
        h, w = image.shape[:2]
        # Resize
        image, ratio, pad = resize(image, self.input_size, self.augment)

        label = self.labels[index].copy()
        if label.size:
            label[:, 1:] = wh2xy(label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])

        nl = len(label)  # number of labels
        h, w = image.shape[:2]
        cls = label[:, 0:1] if nl else np.zeros((0, 1), dtype=np.float32)
        box = label[:, 1:5] if nl else np.zeros((0, 4), dtype=np.float32)
        box = xy2wh(box, w, h)

        if self.augment:
            # Albumentations
            image, box, cls = self.albumentations(image, box, cls)
            nl = len(box)  # update after albumentations
            # Ensure cls is 2D after augmentation
            cls = np.array(cls, dtype=np.float32).reshape(-1, 1) if len(cls) else np.zeros((0, 1), dtype=np.float32)
            # HSV color-space
            augment_hsv(image, self.params)
            # Flip up-down
            if random.random() < self.params.get('flip_ud', 0.0):
                image = np.flipud(image)
                if nl:
                    box[:, 1] = 1 - box[:, 1]
            # Flip left-right
            if random.random() < self.params.get('flip_lr', 0.0):
                image = np.fliplr(image)
                if nl:
                    box[:, 0] = 1 - box[:, 0]

        # Ensure cls and box are tensors
        target_cls = torch.from_numpy(cls).float()
        target_box = torch.from_numpy(box).float()
        # Ensure cls is 2D
        if target_cls.numel() == 0:
            target_cls = torch.zeros((0, 1), dtype=torch.float32)
        elif target_cls.dim() == 1:
            target_cls = target_cls.view(-1, 1)

        # Convert HWC to CHW, BGR to RGB
        sample = image.transpose((2, 0, 1))[::-1]
        sample = np.ascontiguousarray(sample)

        # Ensure indices are consistent
        indices = torch.full((nl,), index, dtype=torch.int64)

        return torch.from_numpy(sample) / 255., target_cls, target_box, indices

    def __len__(self):
        return len(self.filenames)

    def load_image(self, i):
        image = cv2.imread(self.filenames[i])
        if image is None:
            raise FileNotFoundError(f"Image not found: {self.filenames[i]}")
        h, w = image.shape[:2]
        r = self.input_size / max(h, w)
        if r != 1:
            image = cv2.resize(
                image,
                dsize=(int(w * r), int(h * r)),
                interpolation=resample() if self.augment else cv2.INTER_LINEAR
            )
        return image, (h, w)

    @staticmethod
    def collate_fn(batch):
        samples, cls, box, indices = [], [], [], []
        for i, (sample, cl, bx, idx) in enumerate(batch):
            samples.append(sample)
            # Ensure cls is 2D
            if cl.numel() == 0:
                cl = torch.zeros((0, 1), dtype=torch.float32)
            elif cl.dim() == 1:
                cl = cl.view(-1, 1)
            elif cl.dim() > 2:
                cl = cl.squeeze().view(-1, 1)
            cls.append(cl)
            box.append(bx)
            # Update indices to reflect batch position
            idx = torch.full((cl.shape[0],), i, dtype=torch.int64)
            indices.append(idx)

        samples = torch.stack(samples, dim=0).float()
        cls = torch.cat(cls, dim=0) if cls else torch.zeros((0, 1), dtype=torch.float32)
        box = torch.cat(box, dim=0) if box else torch.zeros((0, 4), dtype=torch.float32)
        indices = torch.cat(indices, dim=0) if indices else torch.zeros((0,), dtype=torch.int64)

        targets = {'cls': cls, 'box': box, 'idx': indices}
        return samples, targets

    @staticmethod
    def load_label(filenames):
        x = {}
        for filename in tqdm(filenames, desc="Loading labels"):
            try:
                # Verify images
                with open(filename, 'rb') as f:
                    image = Image.open(f)
                    image.verify()  # PIL verify
                shape = image.size  # image size
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert image.format.lower() in FORMATS, f'invalid image format {image.format}'

                # Verify labels
                a = f'{os.sep}images{os.sep}'
                b = f'{os.sep}labels{os.sep}'
                label_path = b.join(filename.rsplit(a, 1)).rsplit('.', 1)[0] + '.txt'
                if os.path.isfile(label_path):
                    with open(label_path) as f:
                        label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        label = np.array(label, dtype=np.float32)
                    nl = len(label)
                    if nl:
                        assert (label >= 0).all(), f"Negative values in {label_path}"
                        assert label.shape[1] == 5, f"Label format error in {label_path}: expected 5 columns, got {label.shape[1]}"
                        assert (label[:, 1:] <= 1).all(), f"Label coordinates out of bounds in {label_path}"
                        _, i = np.unique(label, axis=0, return_index=True)
                        if len(i) < nl:  # duplicate row check
                            label = label[i]  # remove duplicates
                    else:
                        label = np.zeros((0, 5), dtype=np.float32)
                else:
                    label = np.zeros((0, 5), dtype=np.float32)
            except Exception as e:
                pass
            x[filename] = label
        return x


def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y


def xy2wh(x, w, h):
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1E-3)  # x1, x2
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1E-3)  # y1, y2
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def resample():
    choices = (cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4)
    return random.choice(seq=choices)


def augment_hsv(image, params):
    h = params.get('hsv_h', 0.0)
    s = params.get('hsv_s', 0.0)
    v = params.get('hsv_v', 0.0)
    r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype('uint8')
    lut_s = np.clip(x * r[1], 0, 255).astype('uint8')
    lut_v = np.clip(x * r[2], 0, 255).astype('uint8')
    hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=image)


def resize(image, input_size, augment):
    shape = image.shape[:2]  # current shape [height, width]
    r = min(input_size / shape[0], input_size / shape[1])
    if not augment:
        r = min(r, 1.0)
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2
    if shape[::-1] != pad:
        image = cv2.resize(
            image,
            dsize=pad,
            interpolation=resample() if augment else cv2.INTER_LINEAR
        )
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return image, (r, r), (w, h)


class Albumentations:
    def __init__(self):
        self.transform = A.Compose(
            [
                A.Blur(p=0.01),
                A.CLAHE(p=0.01),
                A.ToGray(p=0.01),
                A.MedianBlur(p=0.01),
            ],
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )

    def __call__(self, image, box, cls):
        if self.transform and len(box) > 0:
            x = self.transform(image=image, bboxes=box, class_labels=cls.flatten())
            image = x['image']
            box = np.array(x['bboxes'])
            cls = np.array(x['class_labels']).reshape(-1, 1) if len(x['class_labels']) else np.zeros((0, 1), dtype=np.float32)
        return image, box, cls


if __name__ == "__main__":
    # Example usage
    dataset = Yolo_Dataset(
        filenames=glob('/content/gdrive/MyDrive/Deep_Learning/Project/slp/test/images/*.jpg'),
        input_size=640,
        params={'flip_ud': 0.5, 'flip_lr': 0.5, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4},
        augment=True
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)

    for images, targets in dataloader:
        print(images.shape, targets)
        break
    print("Dataset loaded successfully.")
