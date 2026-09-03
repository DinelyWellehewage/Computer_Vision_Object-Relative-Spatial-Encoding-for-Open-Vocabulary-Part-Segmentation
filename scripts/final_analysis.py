from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


BASELINE_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "baseline"
)


GEOMETRY_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "geometry"
)


ROBUSTNESS_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "robustness"
)


ALIGNMENT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "alignment"
)


ZOOM_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "object_zoom"
)


FINAL_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_analysis"
)


FINAL_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


def load_json_if_exists(
    path,
):
    if not path.is_file():
        return None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def load_csv_if_exists(
    path,
):
    if not path.is_file():
        return None

    return pd.read_csv(
        path
    )


def print_file_status(
    name,
    path,
):
    status = (
        "FOUND"
        if path.is_file()
        else "MISSING"
    )

    print(
        f"{name:30s}",
        status,
    )

    print(
        " ",
        path,
    )


def main():
    print(
        "Project root:",
        PROJECT_ROOT,
    )

    print(
        "Final output:",
        FINAL_ROOT,
    )

    print()
    print(
        "Experiment files"
    )

    print(
        "----------------"
    )


    expected_files = {

        # Baseline
        "baseline part_only history":
            BASELINE_ROOT
            / "part_only"
            / "history.json",

        "baseline object_mask history":
            BASELINE_ROOT
            / "object_mask"
            / "history.json",


        # Geometry
        "geometry object_mask history":
            GEOMETRY_ROOT
            / "object_mask"
            / "history.json",

        "geometry absolute_xy history":
            GEOMETRY_ROOT
            / "absolute_xy"
            / "history.json",

        "geometry relative_uv history":
            GEOMETRY_ROOT
            / "relative_uv"
            / "history.json",


        # Alignment
        "alignment mask_baseline history":
            ALIGNMENT_ROOT
            / "mask_baseline"
            / "history.json",

        "alignment alignment_mask history":
            ALIGNMENT_ROOT
            / "alignment_mask"
            / "history.json",

        "alignment relative_uv history":
            ALIGNMENT_ROOT
            / "alignment_relative_uv"
            / "history.json",


        # Object zoom
        "zoom mask_baseline history":
            ZOOM_ROOT
            / "mask_baseline"
            / "history.json",

        "zoom alignment_mask history":
            ZOOM_ROOT
            / "alignment_mask"
            / "history.json",

        "zoom relative_uv history":
            ZOOM_ROOT
            / "alignment_relative_uv"
            / "history.json",


        # Robustness
        "robustness results":
            ROBUSTNESS_ROOT
            / "robustness_results.json",
    }


    for name, path in (
        expected_files.items()
    ):
        print_file_status(
            name,
            path,
        )


    print()
    print(
        "Final analysis setup complete."
    )


if __name__ == "__main__":
    main()