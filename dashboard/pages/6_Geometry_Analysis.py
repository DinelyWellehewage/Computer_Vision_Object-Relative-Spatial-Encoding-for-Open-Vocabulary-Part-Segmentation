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
import pandas as pd
import streamlit as st

from dashboard.utils.models import (
    checkpoint_exists,
    load_unseen_dataset,
    predict_sample,
)


st.set_page_config(
    page_title="Geometry Analysis",
    page_icon="📐",
    layout="wide",
)


st.title(
    "Object-Relative Geometry Analysis"
)

st.write(
    """
The geometry representation describes pixel position relative
to the selected parent object rather than relative to the full
image.
"""
)


dataset = load_unseen_dataset()


sample_index = st.number_input(
    "Unseen sample index",
    min_value=0,
    max_value=(
        len(dataset)
        - 1
    ),
    value=10,
    step=1,
)


sample = dataset[
    int(
        sample_index
    )
]


c1, c2, c3 = st.columns(
    3
)

c1.metric(
    "Object",
    sample[
        "object_name"
    ],
)

c2.metric(
    "Part",
    sample[
        "part_name"
    ],
)

c3.metric(
    "Query",
    sample[
        "query"
    ],
)


st.divider()


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


object_mask = (
    sample[
        "crop_object_mask"
    ]
    .squeeze()
    .cpu()
    .numpy()
)


u_map = (
    sample[
        "crop_relative_u"
    ]
    .squeeze()
    .cpu()
    .numpy()
)


v_map = (
    sample[
        "crop_relative_v"
    ]
    .squeeze()
    .cpu()
    .numpy()
)


d_map = (
    sample[
        "crop_boundary_d"
    ]
    .squeeze()
    .cpu()
    .numpy()
)


fig, axes = plt.subplots(
    1,
    5,
    figsize=(
        20,
        4,
    ),
)


axes[0].imshow(
    crop_image
)

axes[0].set_title(
    "Object Crop"
)


axes[1].imshow(
    object_mask
)

axes[1].set_title(
    "Object Mask"
)


axes[2].imshow(
    u_map,
    vmin=0.0,
    vmax=1.0,
)

axes[2].set_title(
    "U — Horizontal"
)


axes[3].imshow(
    v_map,
    vmin=0.0,
    vmax=1.0,
)

axes[3].set_title(
    "V — Vertical"
)


axes[4].imshow(
    d_map,
    vmin=0.0,
    vmax=1.0,
)

axes[4].set_title(
    "D — Boundary Distance"
)


for ax in axes:
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


# ============================================================
# Geometry Interpretation
# ============================================================

st.divider()

st.subheader(
    "Geometry Interpretation"
)


g1, g2, g3 = st.columns(
    3
)


with g1:

    st.markdown(
        """
### U

Normalized horizontal coordinate of each
object pixel relative to the parent object's
bounding box.

- left side of object → small values
- right side of object → large values
"""
    )


with g2:

    st.markdown(
        """
### V

Normalized vertical coordinate of each
object pixel relative to the parent object's
bounding box.

- top of object → small values
- bottom of object → large values
"""
    )


with g3:

    st.markdown(
        """
### D

Normalized distance from the object boundary.

- boundary → near zero
- interior → larger values
"""
    )


# ============================================================
# Fixed vs Query-Gated Geometry
# ============================================================

st.divider()

st.subheader(
    "Fixed vs Query-Gated Geometry"
)

st.write(
    """
The fixed-UVD model always passes all three geometry channels
to the decoder.

The query-gated model predicts query-dependent weights for U,
V, and D from the CLIP text embedding.
"""
)


comparison_modes = {
    "Fixed UVD":
        "alignment_fixed_uvd",

    "Query-Gated UVD":
        "alignment_query_gated_uvd",
}


geometry_comparison = {}


for display_name, comparison_mode in comparison_modes.items():

    if not checkpoint_exists(
        comparison_mode
    ):
        continue

    with st.spinner(
        f"Running {display_name}..."
    ):

        geometry_comparison[
            display_name
        ] = predict_sample(
            sample,
            comparison_mode,
        )


if geometry_comparison:

    columns = st.columns(
        len(
            geometry_comparison
        )
    )


    for column, (
        display_name,
        result,
    ) in zip(
        columns,
        geometry_comparison.items(),
    ):

        with column:

            st.markdown(
                f"### {display_name}"
            )

            st.metric(
                "IoU",
                f"{result['iou']:.4f}",
            )

            st.metric(
                "Dice",
                f"{result['dice']:.4f}",
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


            fig_probability, ax_probability = (
                plt.subplots(
                    figsize=(
                        5,
                        5,
                    )
                )
            )

            ax_probability.imshow(
                probability,
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


# ============================================================
# Query-Gate Interpretation
# ============================================================

st.divider()

st.subheader(
    "Query-Gate Interpretation"
)


gated_result = geometry_comparison.get(
    "Query-Gated UVD"
)


if gated_result is not None:

    gate_weights = (
        gated_result[
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


        geometry_labels = [
            "horizontal position U",
            "vertical position V",
            "boundary distance D",
        ]


        dominant_index = int(
            weights.argmax()
        )


        st.info(
            f"For the query **{sample['query']}**, "
            f"the largest learned geometry weight is assigned "
            f"to **{geometry_labels[dominant_index]}**."
        )


# ============================================================
# Aggregate Query-Gate Analysis
# ============================================================

st.divider()

st.subheader(
    "Average Geometry Weights by Part Query"
)

st.write(
    """
The values below average the learned query-gate weights across
all unseen evaluation samples for each part query. This shows
whether different semantic part queries systematically emphasize
different geometric cues.
"""
)


if not checkpoint_exists(
    "alignment_query_gated_uvd"
):

    st.warning(
        "Query-gated checkpoint is not available "
        "on this machine."
    )

else:

    if st.button(
        "Compute Aggregate Gate Weights"
    ):

        gate_records = []

        progress_bar = st.progress(
            0.0
        )

        status_text = st.empty()

        total_samples = len(
            dataset
        )


        for index in range(
            total_samples
        ):

            current_sample = dataset[
                index
            ]


            status_text.text(
                f"Processing sample "
                f"{index + 1} / {total_samples}"
            )


            result = predict_sample(
                current_sample,
                "alignment_query_gated_uvd",
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


                gate_records.append(
                    {
                        "query":
                            current_sample[
                                "query"
                            ],

                        "part":
                            current_sample[
                                "part_name"
                            ],

                        "U":
                            float(
                                weights[
                                    0
                                ]
                            ),

                        "V":
                            float(
                                weights[
                                    1
                                ]
                            ),

                        "D":
                            float(
                                weights[
                                    2
                                ]
                            ),
                    }
                )


            progress_bar.progress(
                (
                    index
                    + 1
                )
                / total_samples
            )


        status_text.empty()

        progress_bar.empty()


        if gate_records:

            gate_df = pd.DataFrame(
                gate_records
            )


            query_gate_df = (
                gate_df
                .groupby(
                    "query"
                )[
                    [
                        "U",
                        "V",
                        "D",
                    ]
                ]
                .mean()
                .reset_index()
            )


            query_counts = (
                gate_df
                .groupby(
                    "query"
                )
                .size()
                .rename(
                    "Samples"
                )
                .reset_index()
            )


            query_gate_df = (
                query_gate_df.merge(
                    query_counts,
                    on="query",
                )
            )


            geometry_columns = [
                "U",
                "V",
                "D",
            ]


            query_gate_df[
                "Dominant Geometry"
            ] = (
                query_gate_df[
                    geometry_columns
                ]
                .idxmax(
                    axis=1
                )
            )


            query_gate_df = (
                query_gate_df
                .sort_values(
                    "query"
                )
                .reset_index(
                    drop=True
                )
            )


            st.dataframe(
                query_gate_df,
                width="stretch",
                hide_index=True,
            )


            st.markdown(
                "#### Mean Gate Weights"
            )


            chart_df = (
                query_gate_df
                .set_index(
                    "query"
                )[
                    geometry_columns
                ]
            )


            st.bar_chart(
                chart_df
            )


            overall_weights = (
                gate_df[
                    geometry_columns
                ]
                .mean()
            )


            c1, c2, c3 = st.columns(
                3
            )


            c1.metric(
                "Overall mean α U",
                f"{overall_weights['U']:.3f}",
            )

            c2.metric(
                "Overall mean α V",
                f"{overall_weights['V']:.3f}",
            )

            c3.metric(
                "Overall mean α D",
                f"{overall_weights['D']:.3f}",
            )


            dominant_counts = (
                query_gate_df[
                    "Dominant Geometry"
                ]
                .value_counts()
            )


            st.markdown(
                "#### Dominant Cue Across Queries"
            )


            dominant_df = pd.DataFrame(
                {
                    "Geometry":
                        [
                            "U",
                            "V",
                            "D",
                        ],

                    "Number of queries":
                        [
                            int(
                                dominant_counts.get(
                                    "U",
                                    0,
                                )
                            ),

                            int(
                                dominant_counts.get(
                                    "V",
                                    0,
                                )
                            ),

                            int(
                                dominant_counts.get(
                                    "D",
                                    0,
                                )
                            ),
                        ],
                }
            )


            st.bar_chart(
                dominant_df.set_index(
                    "Geometry"
                )
            )


        else:

            st.warning(
                "No gate weights were returned "
                "for the unseen evaluation samples."
            )


# ============================================================
# Interpretation
# ============================================================

st.divider()

st.subheader(
    "Interpretation"
)

st.markdown(
    """
The geometry maps are interpretable, but the final quantitative
evaluation shows that adding U/V/D geometry does not outperform
the simpler object-centric alignment model.

The experiments suggest that:

- **Object-centric cropping** already provides a strong spatial prior.
- **Query alignment** contributes useful semantic localization.
- Explicit **U/V/D geometry** provides only limited additional benefit.
- **Query-conditioned geometry gating** does not improve
  unseen-category generalization under the current configuration.

The strongest final unseen-category model remains
**Object Zoom + Alignment Mask**.
"""
)