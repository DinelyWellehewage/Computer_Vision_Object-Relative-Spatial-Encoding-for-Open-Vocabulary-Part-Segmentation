from pathlib import Path
import sys

import pandas as pd
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


ROBUSTNESS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "robustness"
    / "robustness_results.json"
)


st.set_page_config(
    page_title="Robustness Analysis",
    page_icon="🛡️",
    layout="wide",
)


st.title(
    "Robustness Analysis"
)

st.write(
    """
This page evaluates how explicit spatial representations behave
under object rotation and noisy parent-object masks.

The comparison includes:

- Object-mask conditioning
- Absolute image coordinates (X, Y)
- Object-relative coordinates (U, V)

All robustness experiments are evaluated on the unseen test split.
"""
)


if not ROBUSTNESS_PATH.is_file():

    st.error(
        "Missing robustness results:"
    )

    st.code(
        str(
            ROBUSTNESS_PATH
        )
    )

    st.stop()


df = pd.read_json(
    ROBUSTNESS_PATH
)


DISPLAY_NAMES = {
    "object_mask":
        "Object Mask",

    "absolute_xy":
        "Absolute XY",

    "relative_uv":
        "Relative UV",
}


df[
    "Model"
] = (
    df[
        "mode"
    ]
    .map(
        DISPLAY_NAMES
    )
    .fillna(
        df[
            "mode"
        ]
    )
)


# ============================================================
# Summary
# ============================================================

st.subheader(
    "Key Findings"
)


clean_rows = (
    df[
        (
            (
                df["perturbation"]
                == "rotation"
            )
            & (
                df["value"]
                .astype(str)
                == "0"
            )
        )
    ]
    .sort_values(
        "iou",
        ascending=False,
    )
)


if len(
    clean_rows
) > 0:

    best_clean = (
        clean_rows.iloc[0]
    )


    c1, c2, c3 = st.columns(
        3
    )


    c1.metric(
        "Best Clean Model",
        best_clean[
            "Model"
        ],
    )


    c2.metric(
        "Clean IoU",
        f"{best_clean['iou']:.4f}",
    )


    c3.metric(
        "Clean Dice",
        f"{best_clean['dice']:.4f}",
    )


st.info(
    """
Relative UV consistently performs best across the tested
rotation angles and mask perturbations.

However, all models degrade under severe rotation and strong
mask corruption, so object-relative coordinates improve
robustness but do not make the representation rotation invariant.
"""
)


# ============================================================
# Rotation robustness
# ============================================================

st.divider()

st.subheader(
    "Rotation Robustness"
)

st.write(
    """
The image, parent-object mask, and target part mask are rotated
together. Geometry maps are then recomputed from the rotated
parent-object mask.
"""
)


rotation = (
    df[
        df[
            "perturbation"
        ]
        == "rotation"
    ]
    .copy()
)


rotation[
    "angle"
] = (
    pd.to_numeric(
        rotation[
            "value"
        ]
    )
)


rotation_iou = (
    rotation
    .pivot(
        index="angle",
        columns="Model",
        values="iou",
    )
    .sort_index()
)


rotation_dice = (
    rotation
    .pivot(
        index="angle",
        columns="Model",
        values="dice",
    )
    .sort_index()
)


left, right = st.columns(
    2
)


with left:

    st.markdown(
        "### IoU vs Rotation"
    )

    st.line_chart(
        rotation_iou
    )


with right:

    st.markdown(
        "### Dice vs Rotation"
    )

    st.line_chart(
        rotation_dice
    )


st.markdown(
    "### Rotation Results"
)


rotation_table = (
    rotation[
        [
            "Model",
            "angle",
            "iou",
            "dice",
        ]
    ]
    .sort_values(
        [
            "angle",
            "Model",
        ]
    )
    .rename(
        columns={
            "angle":
                "Angle (°)",

            "iou":
                "IoU",

            "dice":
                "Dice",
        }
    )
)


st.dataframe(
    rotation_table,
    width="stretch",
    hide_index=True,
)


# ============================================================
# Rotation degradation
# ============================================================

st.markdown(
    "### Degradation from 0° to 90°"
)


degradation_rows = []


for model_name in (
    rotation[
        "Model"
    ]
    .unique()
):

    model_rows = (
        rotation[
            rotation[
                "Model"
            ]
            == model_name
        ]
    )

    row_0 = (
        model_rows[
            model_rows[
                "angle"
            ]
            == 0
        ]
    )

    row_90 = (
        model_rows[
            model_rows[
                "angle"
            ]
            == 90
        ]
    )


    if (
        len(
            row_0
        )
        and len(
            row_90
        )
    ):

        iou_0 = (
            row_0.iloc[0][
                "iou"
            ]
        )

        iou_90 = (
            row_90.iloc[0][
                "iou"
            ]
        )


        degradation_rows.append(
            {
                "Model":
                    model_name,

                "IoU at 0°":
                    iou_0,

                "IoU at 90°":
                    iou_90,

                "Absolute Drop":
                    iou_0
                    - iou_90,

                "Relative Drop (%)":
                    (
                        (
                            iou_0
                            - iou_90
                        )
                        / iou_0
                        * 100
                    ),
            }
        )


degradation_df = pd.DataFrame(
    degradation_rows
)


if not degradation_df.empty:

    st.dataframe(
        degradation_df,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# Mask robustness
# ============================================================

st.divider()

st.subheader(
    "Parent-Mask Robustness"
)

st.write(
    """
The RGB image and part target remain unchanged while the provided
parent-object mask is deliberately corrupted.

The geometry maps are recomputed from the corrupted mask, so this
experiment measures sensitivity to errors in parent-object
localization.
"""
)


mask_noise = (
    df[
        df[
            "perturbation"
        ]
        == "mask_noise"
    ]
    .copy()
)


CONDITION_ORDER = [
    "clean",
    "erode_5",
    "erode_11",
    "dilate_5",
    "dilate_11",
    "shift_5",
    "shift_15",
]


CONDITION_NAMES = {
    "clean":
        "Clean",

    "erode_5":
        "Erode 5",

    "erode_11":
        "Erode 11",

    "dilate_5":
        "Dilate 5",

    "dilate_11":
        "Dilate 11",

    "shift_5":
        "Shift 5 px",

    "shift_15":
        "Shift 15 px",
}


mask_noise[
    "condition"
] = pd.Categorical(
    mask_noise[
        "value"
    ],
    categories=CONDITION_ORDER,
    ordered=True,
)


mask_noise[
    "Condition"
] = (
    mask_noise[
        "value"
    ]
    .map(
        CONDITION_NAMES
    )
)


mask_iou = (
    mask_noise
    .pivot(
        index="condition",
        columns="Model",
        values="iou",
    )
    .sort_index()
)


mask_iou.index = [
    CONDITION_NAMES[
        str(
            value
        )
    ]
    for value
    in mask_iou.index
]


mask_dice = (
    mask_noise
    .pivot(
        index="condition",
        columns="Model",
        values="dice",
    )
    .sort_index()
)


mask_dice.index = [
    CONDITION_NAMES[
        str(
            value
        )
    ]
    for value
    in mask_dice.index
]


left, right = st.columns(
    2
)


with left:

    st.markdown(
        "### IoU under Mask Perturbations"
    )

    st.line_chart(
        mask_iou
    )


with right:

    st.markdown(
        "### Dice under Mask Perturbations"
    )

    st.line_chart(
        mask_dice
    )


st.markdown(
    "### Mask Perturbation Results"
)


mask_table = (
    mask_noise[
        [
            "Model",
            "Condition",
            "iou",
            "dice",
        ]
    ]
    .rename(
        columns={
            "iou":
                "IoU",

            "dice":
                "Dice",
        }
    )
)


st.dataframe(
    mask_table,
    width="stretch",
    hide_index=True,
)


# ============================================================
# Comparison of geometry representations
# ============================================================

st.divider()

st.subheader(
    "Absolute vs Object-Relative Coordinates"
)

st.write(
    """
One of the central questions from the project feedback was
whether object-relative coordinates provide a meaningful benefit
over absolute image coordinates.
"""
)


absolute_clean = (
    mask_noise[
        (
            mask_noise[
                "mode"
            ]
            == "absolute_xy"
        )
        & (
            mask_noise[
                "value"
            ]
            == "clean"
        )
    ]
)


relative_clean = (
    mask_noise[
        (
            mask_noise[
                "mode"
            ]
            == "relative_uv"
        )
        & (
            mask_noise[
                "value"
            ]
            == "clean"
        )
    ]
)


if (
    len(
        absolute_clean
    )
    and len(
        relative_clean
    )
):

    absolute_iou = (
        absolute_clean.iloc[0][
            "iou"
        ]
    )

    relative_iou = (
        relative_clean.iloc[0][
            "iou"
        ]
    )


    gain = (
        relative_iou
        - absolute_iou
    )


    c1, c2, c3 = st.columns(
        3
    )


    c1.metric(
        "Absolute XY IoU",
        f"{absolute_iou:.4f}",
    )


    c2.metric(
        "Relative UV IoU",
        f"{relative_iou:.4f}",
    )


    c3.metric(
        "UV Improvement",
        f"{gain:+.4f}",
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
The robustness experiments show three consistent patterns.

### 1. Relative UV is the strongest representation

Object-relative UV coordinates outperform both object-mask-only
conditioning and absolute XY coordinates across all tested
rotation angles and mask perturbations.

### 2. Relative UV is not rotation invariant

Performance gradually decreases as rotation becomes stronger.
The decrease is particularly clear at 90°, showing that
axis-aligned U/V coordinates remain orientation sensitive.

### 3. Missing object regions are particularly harmful

Mask erosion and large spatial shifts substantially reduce
performance. In contrast, moderate mask dilation can slightly
improve IoU, suggesting that retaining additional surrounding
object context is less harmful than removing true object pixels.

Overall, object-relative geometry improves robustness but does not
fully solve rotation sensitivity or dependence on accurate
parent-object masks.
"""
)
