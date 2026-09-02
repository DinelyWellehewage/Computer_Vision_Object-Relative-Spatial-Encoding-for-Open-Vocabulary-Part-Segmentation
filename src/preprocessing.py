import torch
import torch.nn.functional as F

from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(3, 1, 1)


IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(3, 1, 1)


def resize_and_pad_image(
    image,
    target_size=224,
):
    """
    Resize an image while preserving
    its aspect ratio, then pad it to
    a square image.

    Parameters
    ----------
    image:
        uint8 tensor [3, H, W]

    target_size:
        final image size

    Returns
    -------
    display_image:
        float32 tensor
        [3, target_size, target_size]

    normalized_image:
        ImageNet-normalized tensor
        for DINOv2

    resize_info:
        resize and padding information
    """

    image = image.float() / 255.0

    _, h, w = image.shape

    scale = min(
        target_size / h,
        target_size / w,
    )

    new_h = max(
        1,
        round(h * scale),
    )

    new_w = max(
        1,
        round(w * scale),
    )

    resized = TF.resize(
        image,
        [new_h, new_w],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )

    pad_h = target_size - new_h
    pad_w = target_size - new_w

    top = pad_h // 2
    bottom = pad_h - top

    left = pad_w // 2
    right = pad_w - left

    display_image = F.pad(
        resized,
        (
            left,
            right,
            top,
            bottom,
        ),
        value=0.0,
    )

    normalized_image = (
        display_image - IMAGENET_MEAN
    ) / IMAGENET_STD

    resize_info = {
        "original_height": h,
        "original_width": w,
        "resized_height": new_h,
        "resized_width": new_w,
        "scale": scale,
        "pad_top": top,
        "pad_bottom": bottom,
        "pad_left": left,
        "pad_right": right,
    }

    return (
        display_image,
        normalized_image,
        resize_info,
    )