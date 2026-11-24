""" 
Read the manifest file and pads each image to a square using its own backgrounf colour (We are trying to replicat the approach on the SJSU paper)
Then we will resize and augment the data that we have. 
"""

import cv2
import numpy as np 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

MAX_SIDE = 1024 # Reduce ot shrink a huge microscope images first so loading data is fast 

def pad_to_square(img):
    """ Pad to a square with median color of the image border."""
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    side = max(h, w)
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    color = [int(v) for v in np.median(border, axis=0)]

    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def get_transforms(size=224, train=True):
    steps = [A.Resize(size, size)]
    if train:
        # No "up" in spicule, so flips and rotations are safe.
        steps += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(p=0.2),
        ]
    steps += [A.Normalize(), ToTensorV2()] # To match pretrained weights
    return A.Compose(steps)

class SpiculeDataset(Dataset):
    def __init__(self, df, class_to_idx, size=224, train=False):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = get_transforms(size, train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = cv2.imread(row["image_path"])
        if img is None:
            raise FileNotFoundError(f"Could not read the image: {row['image_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = pad_to_square(img)
        img = self.transform(image=img)["image"]
        return img, self.class_to_idx[row["species"]]