import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2

from dataloaders.data_utils.video import FRAME_TRANSFORM

TRAIN_TRANSFORM = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(
            size=(224, 224), scale=(0.875, 1), ratio=(0.9, 1.1), interpolation=cv2.INTER_AREA
        ),
        A.PlanckianJitter(p=0.9),
        A.Blur(blur_limit=3, p=0.2),
        A.ToGray(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ToTensorV2(),
    ]
)


class Transformation:
    def __call__(self, frames, train):
        if train:
            return TRAIN_TRANSFORM(images=np.stack(frames))["images"]
        return torch.stack([FRAME_TRANSFORM(image=frame)["image"] for frame in frames])
