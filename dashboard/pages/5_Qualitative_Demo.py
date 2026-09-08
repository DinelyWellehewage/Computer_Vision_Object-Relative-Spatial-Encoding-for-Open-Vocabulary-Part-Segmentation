from pathlib import Path
import sys


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import matplotlib.pyplot as plt
import streamlit as st

from dashboard.utils.models import (
    QUALITATIVE_MODELS,
    checkpoint_exists,
    load_unseen_dataset,
    predict_sample,
)


st.set_page_config(
    page_title="Qualitative Demo",
    page_icon="🖼️",
    layout="wide",
)


st.title(
    "Qualitative Prediction Demo"
)

st.write(
    """
Select an unseen test sample and a trained object-centric model.

The model receives the object crop, parent-object mask, and text
query and predicts the requested part. The crop prediction is
then projected back into full-image coordinates.
"""
)


dataset = load_unseen_dataset()


controls_left, controls_right = (
    st.columns(2)
)


with controls_left:

    selected_model_name = (
        st.selectbox(
            "Model",
            list(
                QUALITATIVE_MODELS
                .keys()
            ),
        )
    )


with controls_right:

    sample_index = (
        st.number_input(
            "Unseen sample index",
            min_value=0,
            max_value=(
                len(dataset)
                - 1
            ),
            value=10,
            step=1,
        )
    )


mode = (
    QUALITATIVE_MODELS[
        selected_model_name
    ]
)


if not checkpoint_exists(
    mode
):

    st.warning(
        "The checkpoint for this model is not available "
        "on this machine."
    )

    st.code(
        "outputs/object_zoom/"
        f"{mode}/best.pt"
    )

    st.stop()


sample = dataset[
    int(
        sample_index
    )
]


meta1, meta2, meta3 = st.columns(
    3
)

meta1.metric(
    "Parent Object",
    sample[
        "object_name"
    ],
)

meta2.metric(
    "Requested Part",
    sample[
        "part_name"
    ],
)

meta3.metric(
    "Text Query",
    sample[
        "query"
    ],
)


with st.spinner(
    "Running model inference..."
):

    result = predict_sample(
        sample,
        mode,
    )


if result is None:
    st.error(
        "Could not load model."
    )
    st.stop()


st.divider()


metric1, metric2 = st.columns(
    2
)

metric1.metric(
    "Sample IoU",
    f"{result['iou']:.4f}",
)

metric2.metric(
    "Sample Dice",
    f"{result['dice']:.4f}",
)


full_image = (
    sample[
        "full_display_image"
    ]
    .permute(
        1,
        2,
        0,
    )
    .cpu()
    .numpy()
)

full_image = full_image.clip(
    0.0,
    1.0,
)


full_object = (
    sample[
        "full_object_mask"
    ]
    .squeeze()
    .cpu()
    .numpy()
)


full_target = (
    sample[
        "full_part_mask"
    ]
    .squeeze()
    .cpu()
    .numpy()
)


crop_image = (
    sample[
        "crop_display_image"
    ]
    .permute(
        1,
        2,
        0,
    )
    .cpu()
    .numpy()
)

crop_image = crop_image.clip(
    0.0,
    1.0,
)


probability = (
    result[
        "projected_probability"
    ]
    .squeeze()
    .detach()
    .cpu()
    .numpy()
)


prediction = (
    result[
        "prediction"
    ]
    .squeeze()
    .detach()
    .cpu()
    .numpy()
)


fig, axes = plt.subplots(
    2,
    3,
    figsize=(
        15,
        10,
    ),
)


axes[
    0,
    0,
].imshow(
    full_image
)

axes[
    0,
    0,
].set_title(
    "Full Image"
)


axes[
    0,
    1,
].imshow(
    full_object
)

axes[
    0,
    1,
].set_title(
    "Parent Object Mask"
)


axes[
    0,
    2,
].imshow(
    full_target
)

axes[
    0,
    2,
].set_title(
    "Ground Truth"
)


axes[
    1,
    0,
].imshow(
    crop_image
)

axes[
    1,
    0,
].set_title(
    "Object-Centric Crop"
)


axes[
    1,
    1,
].imshow(
    probability,
    vmin=0.0,
    vmax=1.0,
)

axes[
    1,
    1,
].set_title(
    "Prediction Probability"
)


axes[
    1,
    2,
].imshow(
    full_image
)

axes[
    1,
    2,
].imshow(
    prediction,
    alpha=0.5,
)

axes[
    1,
    2,
].set_title(
    "Predicted Part"
)


for ax in axes.flat:

    ax.axis(
        "off"
    )


plt.tight_layout()


st.pyplot(
    fig,
    width="stretch",
)


plt.close(
    fig
)


gate_weights = (
    result[
        "aux"
    ].get(
        "gate_weights"
    )
)


if gate_weights is not None:

    weights = (
        gate_weights[
            0
        ]
        .detach()
        .cpu()
        .numpy()
    )


    st.divider()

    st.subheader(
        "Query-Conditioned Geometry Weights"
    )


    import pandas as pd


    weight_df = pd.DataFrame(
        {
            "Geometry":
                [
                    "U",
                    "V",
                    "D",
                ],

            "Weight":
                weights,
        }
    )


    c1, c2, c3 = st.columns(
        3
    )

    c1.metric(
        "α U",
        f"{weights[0]:.3f}",
    )

    c2.metric(
        "α V",
        f"{weights[1]:.3f}",
    )

    c3.metric(
        "α D",
        f"{weights[2]:.3f}",
    )


    st.bar_chart(
        weight_df.set_index(
            "Geometry"
        )
    )

st.divider()

st.subheader(
    "Side-by-Side Model Comparison"
)

comparison_models = {
    "Alignment Mask":
        "alignment_mask",

    "Fixed UVD":
        "alignment_fixed_uvd",

    "Query-Gated UVD":
        "alignment_query_gated_uvd",
}


comparison_results = {}


for display_name, comparison_mode in comparison_models.items():

    if not checkpoint_exists(
        comparison_mode
    ):
        continue

    with st.spinner(
        f"Running {display_name}..."
    ):

        comparison_results[
            display_name
        ] = predict_sample(
            sample,
            comparison_mode,
        )


if comparison_results:

    columns = st.columns(
        len(
            comparison_results
        )
    )


    for column, (
        display_name,
        comparison_result,
    ) in zip(
        columns,
        comparison_results.items(),
    ):

        prediction_np = (
            comparison_result[
                "prediction"
            ]
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        probability_np = (
            comparison_result[
                "projected_probability"
            ]
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )


        with column:

            st.markdown(
                f"### {display_name}"
            )

            st.metric(
                "IoU",
                f"{comparison_result['iou']:.4f}",
            )

            st.metric(
                "Dice",
                f"{comparison_result['dice']:.4f}",
            )


            fig_probability, ax_probability = (
                plt.subplots(
                    figsize=(
                        5,
                        5,
                    )
                )
            )

            ax_probability.imshow(
                probability_np,
                vmin=0.0,
                vmax=1.0,
            )

            ax_probability.set_title(
                "Probability Map"
            )

            ax_probability.axis(
                "off"
            )

            st.pyplot(
                fig_probability,
                width="stretch",
            )

            plt.close(
                fig_probability
            )


            fig_prediction, ax_prediction = (
                plt.subplots(
                    figsize=(
                        5,
                        5,
                    )
                )
            )

            ax_prediction.imshow(
                full_image
            )

            ax_prediction.imshow(
                prediction_np,
                alpha=0.5,
            )

            ax_prediction.set_title(
                "Prediction Overlay"
            )

            ax_prediction.axis(
                "off"
            )

            st.pyplot(
                fig_prediction,
                width="stretch",
            )

            plt.close(
                fig_prediction
            )