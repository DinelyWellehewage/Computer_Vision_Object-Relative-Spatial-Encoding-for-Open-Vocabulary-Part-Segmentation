from pathlib import Path
import json
import sys

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from datasets import SegmentationDataset

from src.dino_features import (
    get_device,
    load_dino_model,
)

from src.clip_features import (
    load_clip_model,
    extract_clip_features,
)

from src.baseline_model import (
    BaselinePartSegmenter,
)

from src.metrics import (
    segmentation_loss,
    dice_score,
    iou_score,
)


MODE = "part_only"

BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 1

# Keep these small for the local smoke test.
MAX_TRAIN_SAMPLES = 32
MAX_VALIDATION_SAMPLES = 16


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "baseline"
    / MODE
)


def encode_queries(
    clip_model,
    tokenizer,
    queries,
    device,
):
    text_features = []

    for query in queries:
        feature = extract_clip_features(
            clip_model,
            tokenizer,
            query,
            device=device,
        )

        text_features.append(
            feature
        )

    return torch.cat(
        text_features,
        dim=0,
    )


def train_one_epoch(
    model,
    clip_model,
    tokenizer,
    dataloader,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for batch_index, batch in enumerate(
        dataloader,
        start=1,
    ):
        images = batch[
            "image"
        ].to(device)

        object_masks = batch[
            "object_mask"
        ].to(device)

        part_masks = batch[
            "part_mask"
        ].to(device)

        queries = batch[
            "query"
        ]

        text_features = encode_queries(
            clip_model,
            tokenizer,
            queries,
            device,
        )

        optimizer.zero_grad()

        logits = model(
            images,
            text_features,
            object_masks,
        )

        (
            loss,
            bce_loss,
            dice_loss_value,
        ) = segmentation_loss(
            logits,
            part_masks,
        )

        loss.backward()

        optimizer.step()

        batch_dice = dice_score(
            logits.detach(),
            part_masks,
        )

        batch_iou = iou_score(
            logits.detach(),
            part_masks,
        )

        total_loss += loss.item()
        total_dice += batch_dice.item()
        total_iou += batch_iou.item()

        print(
            f"Train "
            f"{batch_index}/"
            f"{len(dataloader)} "
            f"Loss: {loss.item():.4f} "
            f"BCE: {bce_loss.item():.4f} "
            f"Dice loss: "
            f"{dice_loss_value.item():.4f} "
            f"Dice: "
            f"{batch_dice.item():.4f} "
            f"IoU: "
            f"{batch_iou.item():.4f}"
        )

    number_of_batches = len(
        dataloader
    )

    return {
        "loss":
            total_loss
            / number_of_batches,

        "dice":
            total_dice
            / number_of_batches,

        "iou":
            total_iou
            / number_of_batches,
    }


def validate(
    model,
    clip_model,
    tokenizer,
    dataloader,
    device,
):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():

        for batch_index, batch in enumerate(
            dataloader,
            start=1,
        ):
            images = batch[
                "image"
            ].to(device)

            object_masks = batch[
                "object_mask"
            ].to(device)

            part_masks = batch[
                "part_mask"
            ].to(device)

            queries = batch[
                "query"
            ]

            text_features = (
                encode_queries(
                    clip_model,
                    tokenizer,
                    queries,
                    device,
                )
            )

            logits = model(
                images,
                text_features,
                object_masks,
            )

            (
                loss,
                _,
                _,
            ) = segmentation_loss(
                logits,
                part_masks,
            )

            batch_dice = dice_score(
                logits,
                part_masks,
            )

            batch_iou = iou_score(
                logits,
                part_masks,
            )

            total_loss += (
                loss.item()
            )

            total_dice += (
                batch_dice.item()
            )

            total_iou += (
                batch_iou.item()
            )

            print(
                f"Validation "
                f"{batch_index}/"
                f"{len(dataloader)} "
                f"Loss: "
                f"{loss.item():.4f} "
                f"Dice: "
                f"{batch_dice.item():.4f} "
                f"IoU: "
                f"{batch_iou.item():.4f}"
            )

    number_of_batches = len(
        dataloader
    )

    return {
        "loss":
            total_loss
            / number_of_batches,

        "dice":
            total_dice
            / number_of_batches,

        "iou":
            total_iou
            / number_of_batches,
    }


def save_checkpoint(
    model,
    optimizer,
    epoch,
    validation_metrics,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        OUTPUT_DIR
        / "best_model.pt"
    )

    torch.save(
        {
            "epoch":
                epoch,

            "mode":
                MODE,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "validation_metrics":
                validation_metrics,
        },
        checkpoint_path,
    )

    print(
        "Saved checkpoint:",
        checkpoint_path,
    )


def save_history(
    history,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        OUTPUT_DIR
        / "history.json"
    )

    history_path.write_text(
        json.dumps(
            history,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Saved history:",
        history_path,
    )


def main():
    device = get_device()

    print(
        "Device:",
        device,
    )

    print(
        "Mode:",
        MODE,
    )

    full_train_dataset = (
        SegmentationDataset(
            split="train_seen"
        )
    )

    full_validation_dataset = (
        SegmentationDataset(
            split="validation_seen"
        )
    )

    train_dataset = (
        torch.utils.data.Subset(
            full_train_dataset,
            range(
                min(
                    MAX_TRAIN_SAMPLES,
                    len(
                        full_train_dataset
                    ),
                )
            ),
        )
    )

    validation_dataset = (
        torch.utils.data.Subset(
            full_validation_dataset,
            range(
                min(
                    MAX_VALIDATION_SAMPLES,
                    len(
                        full_validation_dataset
                    ),
                )
            ),
        )
    )

    print(
        "Full training samples:",
        len(
            full_train_dataset
        ),
    )

    print(
        "Training samples used:",
        len(
            train_dataset
        ),
    )

    print(
        "Full validation samples:",
        len(
            full_validation_dataset
        ),
    )

    print(
        "Validation samples used:",
        len(
            validation_dataset
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print()
    print(
        "Loading DINOv2..."
    )

    dino_model = load_dino_model(
        device=device
    )

    print(
        "Loading CLIP..."
    )

    clip_model, tokenizer = (
        load_clip_model(
            device=device
        )
    )

    print(
        "Creating baseline model..."
    )

    model = BaselinePartSegmenter(
        dino_encoder=dino_model,
        mode=MODE,
    ).to(device)

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter
            in model.parameters()
            if parameter.requires_grad
        ),
        lr=LEARNING_RATE,
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Trainable parameters:",
        trainable_parameters,
    )

    history = []

    best_validation_iou = -1.0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        print()
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print()
        print(
            "Training"
        )
        print(
            "--------"
        )

        train_metrics = train_one_epoch(
            model,
            clip_model,
            tokenizer,
            train_loader,
            optimizer,
            device,
        )

        print()
        print(
            "Validation"
        )
        print(
            "----------"
        )

        validation_metrics = validate(
            model,
            clip_model,
            tokenizer,
            validation_loader,
            device,
        )

        epoch_result = {
            "epoch":
                epoch,

            "train":
                train_metrics,

            "validation":
                validation_metrics,
        }

        history.append(
            epoch_result
        )

        print()
        print(
            "Epoch results"
        )

        print(
            "-------------"
        )

        print(
            f"Train loss: "
            f"{train_metrics['loss']:.4f}"
        )

        print(
            f"Train Dice: "
            f"{train_metrics['dice']:.4f}"
        )

        print(
            f"Train IoU: "
            f"{train_metrics['iou']:.4f}"
        )

        print(
            f"Validation loss: "
            f"{validation_metrics['loss']:.4f}"
        )

        print(
            f"Validation Dice: "
            f"{validation_metrics['dice']:.4f}"
        )

        print(
            f"Validation IoU: "
            f"{validation_metrics['iou']:.4f}"
        )

        if (
            validation_metrics[
                "iou"
            ]
            > best_validation_iou
        ):
            best_validation_iou = (
                validation_metrics[
                    "iou"
                ]
            )

            save_checkpoint(
                model,
                optimizer,
                epoch,
                validation_metrics,
            )

        save_history(
            history
        )

    print()
    print(
        "Training completed successfully."
    )

    print(
        "Best validation IoU:",
        best_validation_iou,
    )


if __name__ == "__main__":
    main()