from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DASHBOARD_DATA = (
    PROJECT_ROOT
    / "outputs"
    / "dashboard"
)

SUMMARY_PATH = (
    DASHBOARD_DATA
    / "experiment_summary.csv"
)

HISTORY_PATH = (
    DASHBOARD_DATA
    / "epoch_history.csv"
)

TEST_PATH = (
    DASHBOARD_DATA
    / "final_test_results.csv"
)


st.set_page_config(
    page_title=(
        "Open-Vocabulary Part Segmentation"
    ),
    page_icon="🧩",
    layout="wide",
)


st.title(
    "Text-Conditioned Object-Relative Geometry "
    "for Open-Vocabulary Part Segmentation"
)

st.caption(
    "Dinely Shanuka Welle Hewage · "
    "Sahil Rahul Fulfagar · "
    "Sunil Bharatbhai Talaviya"
)


# ============================================================
# Data loading
# ============================================================

@st.cache_data
def load_summary():
    return pd.read_csv(
        SUMMARY_PATH
    )


@st.cache_data
def load_history():
    return pd.read_csv(
        HISTORY_PATH
    )


@st.cache_data
def load_test_results():
    if not TEST_PATH.is_file():
        return None

    return pd.read_csv(
        TEST_PATH
    )


if not SUMMARY_PATH.is_file():
    st.error(
        f"Missing: {SUMMARY_PATH}"
    )
    st.stop()


summary = load_summary()
history = load_history()
test_results = load_test_results()


# ============================================================
# Project overview
# ============================================================

st.header(
    "Project Overview"
)

st.write(
    """
This project investigates whether object-relative geometry
improves text-conditioned part segmentation.

The model combines frozen DINOv2 visual features, frozen CLIP
text embeddings, a parent-object mask, and geometric cues
representing horizontal position (U), vertical position (V),
and normalized distance from the object boundary (D).

The final query-gated model learns how strongly each geometric
cue should contribute for a given part query.
"""
)


# ============================================================
# Research question
# ============================================================

st.header(
    "Research Question"
)

st.info(
    """
Can query-adaptive object-relative geometry improve
open-vocabulary part segmentation by exploiting complementary
horizontal, vertical, and boundary-distance cues, particularly
for unseen parent-object categories?
"""
)



col1, col2, col3, col4 = st.columns(
    4
)

col1.metric(
    "Train samples",
    "32,698",
)

col2.metric(
    "Validation samples",
    "3,708",
)

col3.metric(
    "Test seen",
    "3,371",
)

col4.metric(
    "Test unseen",
    "1,586",
)


# ============================================================
# Experiment ranking
# ============================================================

st.header(
    "Validation Performance"
)

ranking = (
    summary
    .sort_values(
        "best_val_iou",
        ascending=False,
    )
    .copy()
)

ranking[
    "experiment"
] = (
    ranking["family"]
    + " / "
    + ranking["mode"]
)


best_row = ranking.iloc[0]

m1, m2, m3 = st.columns(
    3
)

m1.metric(
    "Best model",
    best_row["mode"],
)

m2.metric(
    "Best validation IoU",
    f"{best_row['best_val_iou']:.4f}",
)

m3.metric(
    "Validation Dice",
    f"{best_row['best_val_dice']:.4f}",
)


st.subheader(
    "Model Ranking"
)

chart_data = (
    ranking[
        [
            "experiment",
            "best_val_iou",
        ]
    ]
    .set_index(
        "experiment"
    )
)

st.bar_chart(
    chart_data
)


display_columns = [
    "family",
    "mode",
    "best_epoch",
    "best_val_iou",
    "best_val_dice",
    "best_val_loss",
    "train_iou_at_best",
    "epochs",
]

st.dataframe(
    ranking[
        display_columns
    ],
    width="stretch",
    hide_index=True,
)


# ============================================================
# Training curves
# ============================================================

st.header(
    "Training Curves"
)

history = history.copy()

history[
    "experiment"
] = (
    history["family"]
    + " / "
    + history["mode"]
)


experiments = sorted(
    history[
        "experiment"
    ].unique()
)


selected_experiment = st.selectbox(
    "Experiment",
    experiments,
)


selected = history[
    history["experiment"]
    == selected_experiment
].sort_values(
    "epoch"
)


st.subheader(
    "IoU"
)

iou_data = (
    selected[
        [
            "epoch",
            "train_iou",
            "val_iou",
        ]
    ]
    .set_index(
        "epoch"
    )
)

st.line_chart(
    iou_data
)


st.subheader(
    "Dice"
)

dice_columns = [
    column
    for column in [
        "train_dice",
        "val_dice",
    ]
    if column in selected.columns
]

if dice_columns:

    dice_data = (
        selected[
            [
                "epoch",
                *dice_columns,
            ]
        ]
        .set_index(
            "epoch"
        )
    )

    st.line_chart(
        dice_data
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
    if column in selected.columns
]

if loss_columns:

    loss_data = (
        selected[
            [
                "epoch",
                *loss_columns,
            ]
        ]
        .set_index(
            "epoch"
        )
    )

    st.line_chart(
        loss_data
    )


# ============================================================
# Final seen / unseen results
# ============================================================

st.header(
    "Seen vs Unseen Generalization"
)

if test_results is None:

    st.info(
        "Final test results are not available yet."
    )

else:

    st.dataframe(
        test_results,
        width="stretch",
        hide_index=True,
    )

    comparison = (
        test_results
        .pivot(
            index="model",
            columns="split",
            values="iou",
        )
        .rename(
            columns={
                "test_seen":
                    "Seen IoU",

                "test_unseen":
                    "Unseen IoU",
            }
        )
    )

    st.subheader(
        "IoU Comparison"
    )

    st.bar_chart(
        comparison
    )


    dice_comparison = (
        test_results
        .pivot(
            index="model",
            columns="split",
            values="dice",
        )
        .rename(
            columns={
                "test_seen":
                    "Seen Dice",

                "test_unseen":
                    "Unseen Dice",
            }
        )
    )

    st.subheader(
        "Dice Comparison"
    )

    st.bar_chart(
        dice_comparison
    )


    unseen_rows = (
        test_results[
            test_results["split"]
            == "test_unseen"
        ]
        .sort_values(
            "iou",
            ascending=False,
        )
    )

    best_unseen = (
        unseen_rows.iloc[0]
    )

    st.success(
        "Best unseen-category model: "
        f"{best_unseen['model']} "
        f"(IoU {best_unseen['iou']:.4f})"
    )

