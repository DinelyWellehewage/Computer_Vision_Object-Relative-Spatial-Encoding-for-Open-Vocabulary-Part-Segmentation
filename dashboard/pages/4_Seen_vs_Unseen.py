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
    load_test_results,
)


st.set_page_config(
    page_title="Seen vs Unseen",
    page_icon="🌍",
    layout="wide",
)


st.title(
    "Seen vs Unseen Generalization"
)

st.write(
    """
Final test performance is reported separately for parent-object
categories seen during training and completely unseen
parent-object categories.
"""
)


results = load_test_results()

if results is None:
    st.error(
        "Missing final_test_results.csv"
    )
    st.stop()


seen = (
    results[
        results["split"]
        == "test_seen"
    ]
    .sort_values(
        "iou",
        ascending=False,
    )
)


unseen = (
    results[
        results["split"]
        == "test_unseen"
    ]
    .sort_values(
        "iou",
        ascending=False,
    )
)


best_seen = seen.iloc[0]
best_unseen = unseen.iloc[0]


c1, c2 = st.columns(
    2
)

c1.metric(
    "Best Seen IoU",
    f"{best_seen['iou']:.4f}",
    best_seen[
        "model"
    ],
)

c2.metric(
    "Best Unseen IoU",
    f"{best_unseen['iou']:.4f}",
    best_unseen[
        "model"
    ],
)


st.divider()


comparison = (
    results
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
    results
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


st.subheader(
    "Complete Results"
)


display = results.copy()

display[
    "split"
] = display[
    "split"
].replace(
    {
        "test_seen":
            "Seen",

        "test_unseen":
            "Unseen",
    }
)


display = display.rename(
    columns={
        "model":
            "Model",

        "split":
            "Split",

        "iou":
            "IoU",

        "dice":
            "Dice",
    }
)


st.dataframe(
    display[
        [
            "Model",
            "Split",
            "IoU",
            "Dice",
        ]
    ],
    width="stretch",
    hide_index=True,
)


st.divider()


st.subheader(
    "Generalization Finding"
)

st.success(
    """
The simpler **Object Zoom + Alignment Mask** model achieves the
highest unseen-category IoU:

**0.3169**

Fixed UVD reaches approximately **0.3091**, while query-gated
UVD reaches approximately **0.3074**.

The experiment therefore does not support the hypothesis that
additional U/V/D gating improves unseen-category generalization
under the current configuration.
"""
)
