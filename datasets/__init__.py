from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .pascal_part116 import (
        PascalPart116Dataset,
    )

    from .segmentation_dataset import (
        SegmentationDataset,
    )


__all__ = [
    "PascalPart116Dataset",
    "SegmentationDataset",
]


def __getattr__(name):
    if name == "PascalPart116Dataset":
        from .pascal_part116 import (
            PascalPart116Dataset,
        )

        return PascalPart116Dataset

    if name == "SegmentationDataset":
        from .segmentation_dataset import (
            SegmentationDataset,
        )

        return SegmentationDataset

    raise AttributeError(
        f"module {__name__!r} "
        f"has no attribute {name!r}"
    )