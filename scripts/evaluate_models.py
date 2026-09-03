from pathlib import Path
import json
import sys

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from datasets import (
    SegmentationDataset,
    GeometryDataset,
    AlignmentDataset,
)

from src.dino_features import (
    get_device,
    load_dino_model,
)

from src.clip_features import (
    load_clip_model,
    extract_clip_features,
)

from src.baseline_model import (
    BaselinePartSegmenter,
)

from src.geometry_model import (
    GeometryPartSegmenter,
)

from src.alignment_model import (
    PartQueryAlignmentSegmenter,
)


DEVICE = get_device()

BATCH_SIZE = 16
NUM_WORKERS = 4

THRESHOLD = 0.5


EXPERIMENTS = [
    {
        "name":
            "baseline_part_only",

        "family":
            "baseline",

        "mode":
            "part_only",

        "checkpoint":
            PROJECT_ROOT
            / "outputs"
            / "baseline"
            / "part_only"
            / "best.pt",
    },

    {
        "name":
            "baseline_object_mask",

        "family":
            "baseline",

        "mode":
            "object_mask",

        "checkpoint":
            PROJECT_ROOT
            / "outputs"
            / "baseline"
            / "object_mask"
            / "best.pt",
    },

    {
        "name":
            "geometry_object_mask",

        "family":
            "geometry",

        "mode":
            "object_mask",

        "checkpoint":
            PROJECT_ROOT
            / "outputs"
            / "geometry"
            / "object_mask"
            / "best.pt",
    },

    {
        "name":
            "geometry_absolute_xy",

        "family":
            "geometry",

        "mode":
            "absolute_xy",

        "checkpoint":
            PROJECT_ROOT
            / "outputs"
            / "geometry"
            / "absolute_xy"
            / "best.pt",
    },

    {
        "name":
            "geometry_relative_uv",

        "family":
            "geometry",

        "mode":
            "relative_uv",

        "checkpoint":
            PROJECT_ROOT
            / "outputs"
            / "geometry"
            / "relative_uv"
            / "best.pt",
    },

    {
        "name":
            "alignment_mask_baseline",

        "family":
            "alignment",

        "mode":
            "mask_baseline",

        "checkpoint":
            PROJECT_ROOT
            / "outputs"
            / "alignment"
            / "mask_baseline"
            / "best.pt",
    },

    {
        "name":
            "alignment_mask",

        "family":
            "alignment",

        "mode":
            "alignment_mask",

        "checkpoint":
            PROJECT_ROOT
            / "outputs"
            / "alignment"
            / "alignment_mask"
            / "best.pt",
    },

    {
        "name":
            "alignment_relative_uv",

        "family":
            "alignment",

        "mode":
            "alignment_relative_uv",

        "checkpoint":
            PROJECT_ROOT
            / "outputs"
            / "alignment"
            / "alignment_relative_uv"
            / "best.pt",
    },
]


def encode_queries(
    clip_model,
    tokenizer,
    queries,
):
    features = []

    for query in queries:
        feature = extract_clip_features(
            clip_model,
            tokenizer,
            query,
            device=DEVICE,
        )

        features.append(
            feature
        )

    return torch.cat(
        features,
        dim=0,
    )


def binary_metrics(
    logits,
    target,
):
    probability = torch.sigmoid(
        logits
    )

    prediction = (
        probability > THRESHOLD
    )

    target = (
        target > THRESHOLD
    )

    intersection = (
        prediction & target
    ).sum(
        dim=(1, 2, 3)
    ).float()

    union = (
        prediction | target
    ).sum(
        dim=(1, 2, 3)
    ).float()

    prediction_sum = (
        prediction
        .sum(
            dim=(1, 2, 3)
        )
        .float()
    )

    target_sum = (
        target
        .sum(
            dim=(1, 2, 3)
        )
        .float()
    )

    iou = (
        intersection
        / union.clamp_min(1.0)
    )

    dice = (
        2.0 * intersection
        / (
            prediction_sum
            + target_sum
        ).clamp_min(1.0)
    )

    return (
        iou,
        dice,
    )


def load_baseline_model(
    experiment,
    dino,
):
    model = BaselinePartSegmenter(
        dino_encoder=dino,
        mode=experiment[
            "mode"
        ],
    ).to(
        DEVICE
    )

    checkpoint = torch.load(
        experiment[
            "checkpoint"
        ],
        map_location=DEVICE,
    )

    state = checkpoint[
        "model_state"
    ]

    model.visual_projection.load_state_dict(
        state[
            "visual_projection"
        ]
    )

    model.text_projection.load_state_dict(
        state[
            "text_projection"
        ]
    )

    model.decoder.load_state_dict(
        state[
            "decoder"
        ]
    )

    model.eval()

    return model


def load_geometry_model(
    experiment,
    dino,
):
    model = GeometryPartSegmenter(
        dino_encoder=dino,
        mode=experiment[
            "mode"
        ],
    ).to(
        DEVICE
    )

    checkpoint = torch.load(
        experiment[
            "checkpoint"
        ],
        map_location=DEVICE,
    )

    state = checkpoint[
        "model_state"
    ]

    model.visual_projection.load_state_dict(
        state[
            "visual_projection"
        ]
    )

    model.text_projection.load_state_dict(
        state[
            "text_projection"
        ]
    )

    model.decoder.load_state_dict(
        state[
            "decoder"
        ]
    )

    model.eval()

    return model


def load_alignment_model(
    experiment,
    dino,
):
    model = PartQueryAlignmentSegmenter(
        dino_encoder=dino,
        mode=experiment[
            "mode"
        ],
    ).to(
        DEVICE
    )

    checkpoint = torch.load(
        experiment[
            "checkpoint"
        ],
        map_location=DEVICE,
    )

    state = checkpoint[
        "model_state"
    ]

    model.visual_projection.load_state_dict(
        state[
            "visual_projection"
        ]
    )

    model.text_projection.load_state_dict(
        state[
            "text_projection"
        ]
    )

    model.text_decoder_projection.load_state_dict(
        state[
            "text_decoder_projection"
        ]
    )

    model.decoder.load_state_dict(
        state[
            "decoder"
        ]
    )

    model.eval()

    return model


def load_model(
    experiment,
    dino,
):
    if experiment[
        "family"
    ] == "baseline":
        return load_baseline_model(
            experiment,
            dino,
        )

    if experiment[
        "family"
    ] == "geometry":
        return load_geometry_model(
            experiment,
            dino,
        )

    return load_alignment_model(
        experiment,
        dino,
    )


def make_dataset(
    family,
    split,
):
    if family == "baseline":
        return SegmentationDataset(
            split=split
        )

    if family == "geometry":
        return GeometryDataset(
            split=split
        )

    return AlignmentDataset(
        split=split
    )


def forward_model(
    model,
    experiment,
    batch,
    text_embeddings,
):
    family = experiment[
        "family"
    ]

    image = batch[
        "image"
    ].to(
        DEVICE,
        non_blocking=True,
    )

    object_mask = batch[
        "object_mask"
    ].to(
        DEVICE,
        non_blocking=True,
    )

    if family == "baseline":
        logits = model(
            image,
            text_embeddings,
            object_mask,
        )

        return logits

    if family == "geometry":
        logits = model(
            image,
            text_embeddings,
            object_mask,

            batch[
                "absolute_x"
            ].to(
                DEVICE,
                non_blocking=True,
            ),

            batch[
                "absolute_y"
            ].to(
                DEVICE,
                non_blocking=True,
            ),

            batch[
                "relative_u"
            ].to(
                DEVICE,
                non_blocking=True,
            ),

            batch[
                "relative_v"
            ].to(
                DEVICE,
                non_blocking=True,
            ),
        )

        return logits

    logits, _ = model(
        image,
        text_embeddings,
        object_mask,

        batch[
            "relative_u"
        ].to(
            DEVICE,
            non_blocking=True,
        ),

        batch[
            "relative_v"
        ].to(
            DEVICE,
            non_blocking=True,
        ),
    )

    return logits


def evaluate(
    model,
    experiment,
    split,
    clip_model,
    tokenizer,
):
    dataset = make_dataset(
        experiment[
            "family"
        ],
        split,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    iou_sum = 0.0
    dice_sum = 0.0

    total_samples = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):
            text_embeddings = (
                encode_queries(
                    clip_model,
                    tokenizer,
                    batch[
                        "query"
                    ],
                )
            )

            logits = forward_model(
                model,
                experiment,
                batch,
                text_embeddings,
            )

            target = batch[
                "part_mask"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            iou, dice = binary_metrics(
                logits,
                target,
            )

            iou_sum += (
                iou.sum().item()
            )

            dice_sum += (
                dice.sum().item()
            )

            total_samples += (
                target.shape[0]
            )

            if (
                batch_index % 100 == 0
                or batch_index
                == len(loader)
            ):
                print(
                    f"{split} "
                    f"{batch_index}/"
                    f"{len(loader)}"
                )

    return {
        "samples":
            total_samples,

        "iou":
            iou_sum
            / total_samples,

        "dice":
            dice_sum
            / total_samples,
    }


def main():
    print(
        "Device:",
        DEVICE,
    )

    print(
        "Loading DINOv2..."
    )

    dino = load_dino_model(
        device=DEVICE
    )

    print(
        "Loading CLIP..."
    )

    clip_model, tokenizer = (
        load_clip_model(
            device=DEVICE
        )
    )

    results = []

    for experiment in EXPERIMENTS:
        checkpoint = experiment[
            "checkpoint"
        ]

        print()
        print(
            "========================================"
        )

        print(
            experiment[
                "name"
            ]
        )

        print(
            "========================================"
        )

        if not checkpoint.is_file():
            print(
                "Checkpoint missing."
            )

            results.append(
                {
                    "experiment":
                        experiment[
                            "name"
                        ],

                    "status":
                        "missing",
                }
            )

            continue

        model = load_model(
            experiment,
            dino,
        )

        row = {
            "experiment":
                experiment[
                    "name"
                ],

            "status":
                "available",
        }

        for split in [
            "test_seen",
            "test_unseen",
        ]:
            print()
            print(
                "Evaluating:",
                split,
            )

            metrics = evaluate(
                model,
                experiment,
                split,
                clip_model,
                tokenizer,
            )

            row[
                f"{split}_iou"
            ] = metrics[
                "iou"
            ]

            row[
                f"{split}_dice"
            ] = metrics[
                "dice"
            ]

            row[
                f"{split}_samples"
            ] = metrics[
                "samples"
            ]

            print(
                "IoU:",
                metrics[
                    "iou"
                ],
            )

            print(
                "Dice:",
                metrics[
                    "dice"
                ],
            )

        results.append(
            row
        )

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "final_analysis"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "test_results.json"
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()