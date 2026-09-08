from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

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


DISPLAY_NAMES = {
    "part_only":
        "Part Only",

    "object_mask":
        "Object Mask",

    "absolute_xy":
        "Absolute XY",

    "relative_uv":
        "Relative UV",

    "mask_baseline":
        "Mask Baseline",

    "alignment_mask":
        "Alignment Mask",

    "alignment_relative_uv":
        "Alignment + Relative UV",

    "fixed_uvd":
        "Fixed UVD",

    "query_gated_uvd":
        "Query-Gated UVD",

    "alignment_fixed_uvd":
        "Alignment + Fixed UVD",

    "alignment_query_gated_uvd":
        "Alignment + Query-Gated UVD",
}


FAMILY_NAMES = {
    "baseline":
        "Baseline",

    "geometry":
        "Geometry",

    "alignment":
        "Alignment",

    "uvd":
        "UVD Geometry",

    "object_zoom":
        "Object-Centric",
}


@st.cache_data
def load_summary():

    if not SUMMARY_PATH.is_file():
        return None

    df = pd.read_csv(
        SUMMARY_PATH
    )

    df["display_name"] = (
        df["mode"]
        .map(
            DISPLAY_NAMES
        )
        .fillna(
            df["mode"]
        )
    )

    df["family_name"] = (
        df["family"]
        .map(
            FAMILY_NAMES
        )
        .fillna(
            df["family"]
        )
    )

    df["experiment_name"] = (
        df["family_name"]
        + " · "
        + df["display_name"]
    )

    return df


@st.cache_data
def load_history():

    if not HISTORY_PATH.is_file():
        return None

    df = pd.read_csv(
        HISTORY_PATH
    )

    df["display_name"] = (
        df["mode"]
        .map(
            DISPLAY_NAMES
        )
        .fillna(
            df["mode"]
        )
    )

    df["family_name"] = (
        df["family"]
        .map(
            FAMILY_NAMES
        )
        .fillna(
            df["family"]
        )
    )

    df["experiment_name"] = (
        df["family_name"]
        + " · "
        + df["display_name"]
    )

    return df


@st.cache_data
def load_test_results():

    if not TEST_PATH.is_file():
        return None

    return pd.read_csv(
        TEST_PATH
    )
