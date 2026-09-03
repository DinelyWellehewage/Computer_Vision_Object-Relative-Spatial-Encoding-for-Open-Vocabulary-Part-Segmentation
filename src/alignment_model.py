import torch
import torch.nn as nn
import torch.nn.functional as F


class PartQueryAlignmentSegmenter(nn.Module):

    def __init__(
        self,
        dino_encoder,
        mode,
        visual_dim=128,
        text_common_dim=128,
        text_decoder_dim=32,
        temperature=0.10,
    ):
        super().__init__()

        valid_modes = {
            "mask_baseline",
            "alignment_mask",
            "alignment_relative_uv",
        }

        if mode not in valid_modes:
            raise ValueError(
                f"Unknown mode: {mode}"
            )

        self.mode = mode
        self.temperature = temperature

        # Frozen DINOv2
        self.dino = dino_encoder

        for parameter in self.dino.parameters():
            parameter.requires_grad = False

        # DINO: 384 -> 128
        self.visual_projection = nn.Conv2d(
            384,
            visual_dim,
            kernel_size=1,
        )

        # CLIP: 512 -> 128
        self.text_projection = nn.Sequential(
            nn.Linear(
                512,
                text_common_dim,
            ),
            nn.GELU(),
        )

        # Text for decoder:
        # 128 -> 32
        self.text_decoder_projection = (
            nn.Linear(
                text_common_dim,
                text_decoder_dim,
            )
        )

        # Fusion channels:
        #
        # visual     128
        # text        32
        # object       1
        # U,V          2
        # alignment    1
        # ----------------
        # total       164

        fusion_dim = (
            visual_dim
            + text_decoder_dim
            + 1
            + 2
            + 1
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                fusion_dim,
                128,
                kernel_size=3,
                padding=1,
            ),

            nn.GroupNorm(
                8,
                128,
            ),

            nn.GELU(),

            nn.Conv2d(
                128,
                64,
                kernel_size=3,
                padding=1,
            ),

            nn.GroupNorm(
                8,
                64,
            ),

            nn.GELU(),

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1,
            ),

            nn.GELU(),

            nn.Conv2d(
                32,
                1,
                kernel_size=1,
            ),
        )


    def train(
        self,
        mode=True,
    ):
        super().train(mode)

        self.dino.eval()

        return self


    @torch.no_grad()
    def extract_dino_features(
        self,
        images,
    ):
        output = self.dino.forward_features(
            images
        )

        tokens = output[
            "x_norm_patchtokens"
        ]

        batch_size = tokens.shape[0]
        num_patches = tokens.shape[1]
        channels = tokens.shape[2]

        grid_size = int(
            num_patches ** 0.5
        )

        features = (
            tokens
            .transpose(
                1,
                2,
            )
            .reshape(
                batch_size,
                channels,
                grid_size,
                grid_size,
            )
        )

        return features


    def compute_alignment(
        self,
        visual,
        text_common,
    ):
        visual_normalized = F.normalize(
            visual,
            dim=1,
        )

        text_normalized = F.normalize(
            text_common,
            dim=1,
        )

        text_spatial = (
            text_normalized[
                :,
                :,
                None,
                None,
            ]
        )

        cosine_similarity = (
            visual_normalized
            * text_spatial
        ).sum(
            dim=1,
            keepdim=True,
        )

        alignment_logits = (
            cosine_similarity
            / self.temperature
        )

        alignment_probability = (
            torch.sigmoid(
                alignment_logits
            )
        )

        return (
            alignment_logits,
            alignment_probability,
        )


    def forward(
        self,
        images,
        text_embeddings,
        object_mask,
        relative_u,
        relative_v,
    ):
        # DINO features
        dino_features = (
            self.extract_dino_features(
                images
            )
        )

        # [B,128,16,16]
        visual = self.visual_projection(
            dino_features
        )

        # CLIP text -> shared 128-D space
        text_common = self.text_projection(
            text_embeddings
        )

        # Text for decoder -> 32-D
        text_decoder = (
            self.text_decoder_projection(
                text_common
            )
        )

        text_map = (
            text_decoder[
                :,
                :,
                None,
                None,
            ]
            .expand(
                -1,
                -1,
                visual.shape[-2],
                visual.shape[-1],
            )
        )

        # Query alignment
        (
            alignment_logits,
            alignment_probability,
        ) = self.compute_alignment(
            visual,
            text_common,
        )

        spatial_size = (
            visual.shape[-2:]
        )

        # Parent object mask
        mask_low = F.interpolate(
            object_mask,
            size=spatial_size,
            mode="nearest",
        )

        # Only keep alignment inside
        # the parent object
        alignment_map = (
            alignment_probability
            * mask_low
        )

        # Relative U/V
        u_low = F.interpolate(
            relative_u,
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )

        v_low = F.interpolate(
            relative_v,
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )

        u_low = (
            u_low
            * mask_low
        )

        v_low = (
            v_low
            * mask_low
        )

        # Controlled modes
        if self.mode == "mask_baseline":
            alignment_input = (
                torch.zeros_like(
                    alignment_map
                )
            )

            u_input = (
                torch.zeros_like(
                    u_low
                )
            )

            v_input = (
                torch.zeros_like(
                    v_low
                )
            )

        elif self.mode == "alignment_mask":
            alignment_input = (
                alignment_map
            )

            u_input = (
                torch.zeros_like(
                    u_low
                )
            )

            v_input = (
                torch.zeros_like(
                    v_low
                )
            )

        else:
            alignment_input = (
                alignment_map
            )

            u_input = u_low
            v_input = v_low

        fused = torch.cat(
            [
                visual,
                text_map,
                mask_low,
                u_input,
                v_input,
                alignment_input,
            ],
            dim=1,
        )

        logits_low = self.decoder(
            fused
        )

        logits = F.interpolate(
            logits_low,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        aux = {
            "alignment_logits":
                alignment_logits,

            "alignment_probability":
                alignment_probability,

            "alignment_map":
                alignment_map,

            "object_mask_low":
                mask_low,
        }

        return (
            logits,
            aux,
        )