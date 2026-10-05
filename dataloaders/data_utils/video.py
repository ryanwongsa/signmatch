import albumentations as A
import cv2
import torch
from albumentations.pytorch.transforms import ToTensorV2

FRAME_TRANSFORM = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ToTensorV2(),
    ]
)


def read_video(path):
    """-> (BGR uint8 frames as OpenCV decodes them, frames per second)."""
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames, fps


def iter_frames(path):
    """BGR uint8 frames as OpenCV decodes them, one at a time."""
    capture = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()


def to_model_input(frames):
    """BGR frames -> (T, 3, 224, 224) float32, RGB and ImageNet-normalised."""
    rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    return torch.stack([FRAME_TRANSFORM(image=frame)["image"] for frame in rgb])
