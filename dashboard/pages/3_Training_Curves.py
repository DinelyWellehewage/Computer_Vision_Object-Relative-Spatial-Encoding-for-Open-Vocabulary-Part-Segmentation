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


import streamlit as st

from dashboard.utils.data import (
    load_history,
)


st.set_page_config(
    page_title="Training Curves",
    page_icon="📈",
    layout="wide",
)


st.title(
    "Training Curves"
)


history = load_history()

if history is None:
    st.error(
        "Missing epoch_history.csv"
    )
    st.stop()


experiments = sorted(
    history[
        "experiment_name"
    ].unique()
)


selected_experiment = st.selectbox(
    "Experiment",
    experiments,
)


selected = (
    history[
        history[
            "experiment_name"
        ]
        == selected_experiment
    ]
    .sort_values(
        "epoch"
    )
)


best_row = selected.loc[
    selected[
        "val_iou"
    ].idxmax()
]


c1, c2, c3, c4 = st.columns(
    4
)

c1.metric(
    "Best Epoch",
    int(
        best_row["epoch"]
    ),
)

c2.metric(
    "Best Val IoU",
    f"{best_row['val_iou']:.4f}",
)

c3.metric(
    "Val Dice",
    f"{best_row['val_dice']:.4f}",
)

c4.metric(
    "Train IoU",
    f"{best_row['train_iou']:.4f}",
)


st.divider()


left, right = st.columns(
    2
)


with left:

    st.subheader(
        "IoU"
    )

    st.line_chart(
        selected[
            [
                "epoch",
                "train_iou",
                "val_iou",
            ]
        ].set_index(
            "epoch"
        )
    )


with right:

    st.subheader(
        "Dice"
    )

    dice_columns = [
        column
        for column in [
            "train_dice",
            "val_dice",
        ]
        if column
        in selected.columns
    ]

    if dice_columns:

        st.line_chart(
            selected[
                [
                    "epoch",
                    *dice_columns,
                ]
            ].set_index(
                "epoch"
            )
        )


st.subheader(
    "Loss"
)

loss_columns = [
    column
    for column in [
        "train_loss",
        "val_loss",
    ]
    if column
    in selected.columns
]


if loss_columns:

    st.line_chart(
        selected[
            [
                "epoch",
                *loss_columns,
            ]
        ].set_index(
            "epoch"
        )
    )


if (
    "learning_rate"
    in selected.columns
):

    st.subheader(
        "Learning Rate"
    )

    st.line_chart(
        selected[
            [
                "epoch",
                "learning_rate",
            ]
        ].set_index(
            "epoch"
        )
    )
    