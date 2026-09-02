from pathlib import Path
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


BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 1

# Only for a quick local smoke test.
MAX_TRAIN_SAMPLES = 32


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

        text_features = torch.cat(
            text_features,
            dim=0,
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
            f"Batch {batch_index}/"
            f"{len(dataloader)} "
            f"Loss: {loss.item():.4f} "
            f"BCE: {bce_loss.item():.4f} "
            f"Dice loss: "
            f"{dice_loss_value.item():.4f} "
            f"Dice: {batch_dice.item():.4f} "
            f"IoU: {batch_iou.item():.4f}"
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


def main():
    device = get_device()

    print(
        "Device:",
        device,
    )

    full_train_dataset = (
        SegmentationDataset(
            split="train_seen"
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

    print(
        "Full training samples:",
        len(
            full_train_dataset
        ),
    )

    print(
        "Smoke-test samples:",
        len(
            train_dataset
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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
        mode="part_only",
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

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        print()
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        metrics = train_one_epoch(
            model,
            clip_model,
            tokenizer,
            train_loader,
            optimizer,
            device,
        )

        print()
        print(
            "Epoch results"
        )

        print(
            "-------------"
        )

        print(
            f"Training loss: "
            f"{metrics['loss']:.4f}"
        )

        print(
            f"Training Dice: "
            f"{metrics['dice']:.4f}"
        )

        print(
            f"Training IoU: "
            f"{metrics['iou']:.4f}"
        )

    print()
    print(
        "Baseline smoke test "
        "completed successfully."
    )


if __name__ == "__main__":
    main()