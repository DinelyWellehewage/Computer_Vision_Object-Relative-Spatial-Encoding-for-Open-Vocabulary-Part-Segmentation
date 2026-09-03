from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .pascal_part116 import (
        PascalPart116Dataset,
    )

    from .segmentation_dataset import (
        SegmentationDataset,
    )

    from .geometry_dataset import (
        GeometryDataset,
    )


__all__ = [
    "PascalPart116Dataset",
    "SegmentationDataset",
    "GeometryDataset",
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

    if name == "GeometryDataset":
        from .geometry_dataset import (
            GeometryDataset,
        )

        return GeometryDataset

    raise AttributeError(
        f"module {__name__!r} "
        f"has no attribute {name!r}"
    )