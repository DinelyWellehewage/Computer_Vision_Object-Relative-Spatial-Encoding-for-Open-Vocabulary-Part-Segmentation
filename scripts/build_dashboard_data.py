from pathlib import Path
import csv
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = PROJECT_ROOT / "outputs"

DASHBOARD_DIR = OUTPUT_ROOT / "dashboard"

DASHBOARD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


EXPERIMENTS = [
    # Baselines
    (
        "baseline",
        "part_only",
        OUTPUT_ROOT
        / "experiments"
        / "part_only"
        / "history.json",
    ),
    (
        "baseline",
        "object_mask",
        OUTPUT_ROOT
        / "experiments"
        / "object_mask"
        / "history.json",
    ),

    # Geometry
    (
        "geometry",
        "object_mask",
        OUTPUT_ROOT
        / "geometry"
        / "object_mask"
        / "history.json",
    ),
    (
        "geometry",
        "absolute_xy",
        OUTPUT_ROOT
        / "geometry"
        / "absolute_xy"
        / "history.json",
    ),
    (
        "geometry",
        "relative_uv",
        OUTPUT_ROOT
        / "geometry"
        / "relative_uv"
        / "history.json",
    ),

    # Alignment
    (
        "alignment",
        "mask_baseline",
        OUTPUT_ROOT
        / "experiments"
        / "part_query_alignment"
        / "mask_baseline"
        / "history.json",
    ),
    (
        "alignment",
        "alignment_mask",
        OUTPUT_ROOT
        / "experiments"
        / "part_query_alignment"
        / "alignment_mask"
        / "history.json",
    ),
    (
        "alignment",
        "alignment_relative_uv",
        OUTPUT_ROOT
        / "experiments"
        / "part_query_alignment"
        / "alignment_relative_uv"
        / "history.json",
    ),

    # Full-image UVD
    (
        "uvd",
        "fixed_uvd",
        OUTPUT_ROOT
        / "experiments"
        / "query_gated_uvd"
        / "fixed_uvd"
        / "history.json",
    ),
    (
        "uvd",
        "query_gated_uvd",
        OUTPUT_ROOT
        / "experiments"
        / "query_gated_uvd"
        / "query_gated_uvd"
        / "history.json",
    ),

    # Object-centric crop
    (
        "object_zoom",
        "mask_baseline",
        OUTPUT_ROOT
        / "object_zoom"
        / "mask_baseline"
        / "history.json",
    ),
    (
        "object_zoom",
        "alignment_mask",
        OUTPUT_ROOT
        / "object_zoom"
        / "alignment_mask"
        / "history.json",
    ),
    (
        "object_zoom",
        "alignment_relative_uv",
        OUTPUT_ROOT
        / "object_zoom"
        / "alignment_relative_uv"
        / "history.json",
    ),
    (
        "object_zoom",
        "alignment_fixed_uvd",
        OUTPUT_ROOT
        / "object_zoom"
        / "alignment_fixed_uvd"
        / "history.json",
    ),
    (
        "object_zoom",
        "alignment_query_gated_uvd",
        OUTPUT_ROOT
        / "object_zoom"
        / "alignment_query_gated_uvd"
        / "history.json",
    ),
]


summary_rows = []
epoch_rows = []


for family, mode, history_path in EXPERIMENTS:

    if not history_path.is_file():
        print(
            "Missing:",
            history_path,
        )
        continue

    history = json.loads(
        history_path.read_text()
    )

    if not history:
        print(
            "Empty history:",
            history_path,
        )
        continue

    best = max(
        history,
        key=lambda row: row["val_iou"],
    )

    runtime_seconds = sum(
        float(
            row.get(
                "seconds",
                0.0,
            )
        )
        for row in history
    )

    summary_rows.append(
        {
            "family":
                family,

            "mode":
                mode,

            "best_epoch":
                best["epoch"],

            "best_val_iou":
                best["val_iou"],

            "best_val_dice":
                best.get(
                    "val_dice"
                ),

            "best_val_loss":
                best.get(
                    "val_loss"
                ),

            "train_iou_at_best":
                best.get(
                    "train_iou"
                ),

            "train_loss_at_best":
                best.get(
                    "train_loss"
                ),

            "runtime_seconds":
                runtime_seconds,

            "epochs":
                len(history),
        }
    )

    for row in history:

        epoch_row = {
            "family":
                family,

            "mode":
                mode,
        }

        epoch_row.update(
            row
        )

        epoch_rows.append(
            epoch_row
        )


def write_csv(
    path,
    rows,
):
    if not rows:
        print(
            "No rows for:",
            path,
        )
        return

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


summary_path = (
    DASHBOARD_DIR
    / "experiment_summary.csv"
)

history_path = (
    DASHBOARD_DIR
    / "epoch_history.csv"
)


write_csv(
    summary_path,
    summary_rows,
)

write_csv(
    history_path,
    epoch_rows,
)


# ============================================================
# Existing final test evaluation
# ============================================================

test_source = (
    OUTPUT_ROOT
    / "object_zoom_evaluation"
    / "object_zoom_results.json"
)

test_output = (
    DASHBOARD_DIR
    / "final_test_results.csv"
)


if test_source.is_file():

    raw_test = json.loads(
        test_source.read_text()
    )

    test_rows = []

    for row in raw_test:

        split = row.get(
            "split",
            "unknown",
        )

        full = row.get(
            "full_image",
            {},
        )

        crop = row.get(
            "object_crop",
            {},
        )

        test_rows.append(
            {
                "split":
                    split,

                "full_image_iou":
                    full.get("iou"),

                "full_image_dice":
                    full.get("dice"),

                "object_crop_iou":
                    crop.get("iou"),

                "object_crop_dice":
                    crop.get("dice"),

                "crop_iou_gain":
                    (
                        crop.get("iou")
                        - full.get("iou")
                        if (
                            crop.get("iou")
                            is not None
                            and full.get("iou")
                            is not None
                        )
                        else None
                    ),
            }
        )

    write_csv(
        test_output,
        test_rows,
    )

else:
    print(
        "Missing final test results:",
        test_source,
    )


print()
print("Created:")
print(summary_path)
print(history_path)

if test_output.is_file():
    print(test_output)
