from pathlib import Path

import torch
import streamlit as st


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


from datasets import (
    ObjectCentricDataset,
)

from src.dino_features import (
    get_device,
    load_dino_model,
)

from src.clip_features import (
    load_clip_model,
    extract_clip_features,
)

from src.alignment_model import (
    PartQueryAlignmentSegmenter,
)

from src.crop_projection import (
    project_crop_prediction_to_full_view,
)


DEVICE = get_device()


QUALITATIVE_MODELS = {
    "Crop + Alignment":
        "alignment_mask",

    "Crop + Alignment + Relative UV":
        "alignment_relative_uv",

    "Crop + Alignment + Fixed UVD":
        "alignment_fixed_uvd",

    "Crop + Alignment + Query-Gated UVD":
        "alignment_query_gated_uvd",
}


UVD_MODES = {
    "alignment_fixed_uvd",
    "alignment_query_gated_uvd",
}


@st.cache_resource
def load_unseen_dataset():

    return ObjectCentricDataset(
        split="test_unseen",
        image_size=224,
        context_ratio=0.15,
    )


@st.cache_resource
def load_seen_dataset():

    return ObjectCentricDataset(
        split="test_seen",
        image_size=224,
        context_ratio=0.15,
    )


@st.cache_resource
def load_encoders():

    dino = load_dino_model(
        device=DEVICE
    )

    clip_model, tokenizer = (
        load_clip_model(
            device=DEVICE
        )
    )

    return (
        dino,
        clip_model,
        tokenizer,
    )


def checkpoint_path(
    mode,
):

    return (
        PROJECT_ROOT
        / "outputs"
        / "object_zoom"
        / mode
        / "best.pt"
    )


def checkpoint_exists(
    mode,
):

    return checkpoint_path(
        mode
    ).is_file()


@st.cache_resource
def load_segmentation_model(
    mode,
):

    path = checkpoint_path(
        mode
    )

    if not path.is_file():
        return None

    (
        dino,
        _,
        _,
    ) = load_encoders()

    model = (
        PartQueryAlignmentSegmenter(
            dino_encoder=dino,
            mode=mode,
        )
        .to(
            DEVICE
        )
    )

    checkpoint = torch.load(
        path,
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

    if (
        model.geometry_gate
        is not None
        and "geometry_gate"
        in state
    ):
        model.geometry_gate.load_state_dict(
            state[
                "geometry_gate"
            ]
        )

    model.eval()

    return model


def encode_query(
    query,
):

    (
        _,
        clip_model,
        tokenizer,
    ) = load_encoders()

    return extract_clip_features(
        clip_model,
        tokenizer,
        query,
        device=DEVICE,
    )


def predict_sample(
    sample,
    mode,
):

    model = load_segmentation_model(
        mode
    )

    if model is None:
        return None

    text = encode_query(
        sample[
            "query"
        ]
    )


    crop_image = (
        sample[
            "crop_image"
        ]
        .unsqueeze(0)
        .to(
            DEVICE
        )
    )

    crop_object = (
        sample[
            "crop_object_mask"
        ]
        .unsqueeze(0)
        .to(
            DEVICE
        )
    )

    crop_u = (
        sample[
            "crop_relative_u"
        ]
        .unsqueeze(0)
        .to(
            DEVICE
        )
    )

    crop_v = (
        sample[
            "crop_relative_v"
        ]
        .unsqueeze(0)
        .to(
            DEVICE
        )
    )


    with torch.no_grad():

        if mode in UVD_MODES:

            crop_d = (
                sample[
                    "crop_boundary_d"
                ]
                .unsqueeze(0)
                .to(
                    DEVICE
                )
            )

            logits, aux = model(
                crop_image,
                text,
                crop_object,
                crop_u,
                crop_v,
                crop_d,
            )

        else:

            logits, aux = model(
                crop_image,
                text,
                crop_object,
                crop_u,
                crop_v,
            )


    crop_probability = (
        torch.sigmoid(
            logits
        )[0]
    )


    projected_probability = (
        project_crop_prediction_to_full_view(
            crop_probability,

            original_height=int(
                sample[
                    "original_height"
                ]
            ),

            original_width=int(
                sample[
                    "original_width"
                ]
            ),

            crop_x1=int(
                sample[
                    "crop_x1"
                ]
            ),

            crop_y1=int(
                sample[
                    "crop_y1"
                ]
            ),

            crop_side=int(
                sample[
                    "crop_side"
                ]
            ),

            target_size=224,
        )
    )


    prediction = (
        projected_probability
        > 0.5
    )


    target = (
        sample[
            "full_part_mask"
        ]
        > 0.5
    )


    intersection = (
        prediction
        & target
    ).sum().float()


    union = (
        prediction
        | target
    ).sum().float()


    prediction_sum = (
        prediction
        .sum()
        .float()
    )

    target_sum = (
        target
        .sum()
        .float()
    )


    iou = (
        intersection
        / union.clamp_min(
            1.0
        )
    ).item()


    dice = (
        2.0
        * intersection
        / (
            prediction_sum
            + target_sum
        ).clamp_min(
            1.0
        )
    ).item()


    return {
        "crop_probability":
            crop_probability,

        "projected_probability":
            projected_probability,

        "prediction":
            prediction.float(),

        "target":
            target.float(),

        "iou":
            iou,

        "dice":
            dice,

        "aux":
            aux,
    }
