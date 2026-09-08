from pathlib import Path
import sys

import streamlit as st


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


from dashboard.utils.data import (
    load_summary,
)


st.set_page_config(
    page_title="Experiment Results",
    page_icon="📊",
    layout="wide",
)


st.title(
    "Experiment Results"
)

st.caption(
    "Validation performance across all completed experiments."
)


summary = load_summary()

if summary is None:
    st.error(
        "Missing experiment_summary.csv"
    )
    st.stop()


ranking = (
    summary
    .sort_values(
        "best_val_iou",
        ascending=False,
    )
    .copy()
)


best = ranking.iloc[0]


c1, c2, c3, c4 = st.columns(
    4
)

c1.metric(
    "Best Model",
    best[
        "display_name"
    ],
)

c2.metric(
    "Validation IoU",
    f"{best['best_val_iou']:.4f}",
)

c3.metric(
    "Validation Dice",
    f"{best['best_val_dice']:.4f}",
)

c4.metric(
    "Best Epoch",
    int(
        best["best_epoch"]
    ),
)


st.divider()


family_options = [
    "All",
    *sorted(
        ranking[
            "family_name"
        ].unique()
    ),
]


selected_family = st.selectbox(
    "Experiment family",
    family_options,
)


if selected_family != "All":

    filtered = ranking[
        ranking[
            "family_name"
        ]
        == selected_family
    ].copy()

else:

    filtered = ranking.copy()


st.subheader(
    "Validation IoU Ranking"
)


chart = (
    filtered[
        [
            "experiment_name",
            "best_val_iou",
        ]
    ]
    .set_index(
        "experiment_name"
    )
)


st.bar_chart(
    chart
)


st.subheader(
    "Experiment Table"
)


table = filtered[
    [
        "family_name",
        "display_name",
        "best_epoch",
        "best_val_iou",
        "best_val_dice",
        "best_val_loss",
        "train_iou_at_best",
        "runtime_seconds",
    ]
].copy()


table[
    "runtime_minutes"
] = (
    table[
        "runtime_seconds"
    ]
    / 60
)


table[
    "runtime_minutes"
] = (
    table[
        "runtime_minutes"
    ].round(1)
)


table = table.drop(
    columns=[
        "runtime_seconds"
    ]
)


table = table.rename(
    columns={
        "family_name":
            "Family",

        "display_name":
            "Model",

        "best_epoch":
            "Best Epoch",

        "best_val_iou":
            "Val IoU",

        "best_val_dice":
            "Val Dice",

        "best_val_loss":
            "Val Loss",

        "train_iou_at_best":
            "Train IoU",

        "runtime_minutes":
            "Runtime (min)",
    }
)


st.dataframe(
    table,
    width="stretch",
    hide_index=True,
)


st.divider()


st.subheader(
    "Main Validation Finding"
)

st.success(
    """
The strongest validation model is the object-centric
**Alignment Mask** variant with validation IoU approximately
**0.3878**.

Object-centric cropping provides a much larger improvement than
the additional fixed or query-gated UVD geometry.
"""
)
