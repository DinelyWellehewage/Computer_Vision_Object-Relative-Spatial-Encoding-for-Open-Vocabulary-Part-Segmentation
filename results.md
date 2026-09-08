# Experimental Results

## Experimental Protocol

All models are trained on `train_seen` and model selection is performed using `validation_seen`.

| Split | Samples |
|---|---:|
| Train seen | 32,698 |
| Validation seen | 3,708 |
| Test seen | 3,371 |
| Test unseen | 1,586 |

The best checkpoint for each experiment is selected using validation IoU. Test sets are reserved for final evaluation.

---

## 1. Baseline Experiments

### 1.1 Part-Only Baseline

**Status:** Completed

| Setting | Value |
|---|---|
| Mode | `part_only` |
| Epochs | 20 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| GPU | NVIDIA A40 |
| AMP | Enabled |
| Training samples | 32,698 |
| Validation samples | 3,708 |

### Results

| Metric | Result |
|---|---:|
| Best validation IoU | **0.3083** |
| Validation Dice at best checkpoint | **0.3955** |
| Final training IoU | 0.4075 |
| Final training loss | 0.5389 |
| Final validation loss | 0.6672 |
| Best epoch | 20 |
| Total epochs | 20 |
| Training runtime | ~1 h 27 min |

Validation IoU improved from **0.2098** at epoch 1 to **0.3083** at epoch 20.

The best validation checkpoint was obtained at epoch 20.

Artifacts:

- `outputs/experiments/part_only/best.pt`
- `outputs/experiments/part_only/last.pt`
- `outputs/experiments/part_only/history.json`
- `outputs/experiments/part_only/config.json`

---

### 1.2 Object-Mask Baseline

**Status:** Completed

| Metric | Result |
|---|---:|
| Best validation IoU | **0.3210** |
| Final validation IoU | 0.3170 |
| Final validation Dice | 0.4040 |
| Final training IoU | 0.4372 |
| Final training loss | 0.5053 |
| Final validation loss | 0.6562 |
| Total epochs | 20 |
| Training runtime | ~1 h 27 min |

The object-mask baseline outperformed the part-only baseline in validation IoU.

Best validation IoU improved from **0.3083** to **0.3210**.

Artifacts:

- `outputs/experiments/object_mask/best.pt`
- `outputs/experiments/object_mask/last.pt`
- `outputs/experiments/object_mask/history.json`
- `outputs/experiments/object_mask/config.json`

---

## 2. Geometry Experiments

| Method | Best Val IoU | Val Dice | Status |
|---|---:|---:|---|
| Object mask | **0.3246** | 0.4141 | Completed |
| Absolute XY | **0.3291** | 0.4101 | Completed |
| Object-relative UV | **0.3301** | 0.4082 | Completed |

---


### 2.1 Geometry Object-Mask

**Status:** Completed

| Metric | Result |
|---|---:|
| Best validation IoU | **0.3246** |
| Final validation IoU | 0.3148 |
| Final validation Dice | 0.4012 |
| Final training IoU | 0.4369 |
| Final training loss | 0.5042 |
| Final validation loss | 0.6577 |
| Total epochs | 20 |
| Training runtime | ~1 h 30 min |

The best validation IoU was achieved at epoch 18:

- Validation IoU: **0.3246**
- Validation Dice: **0.4141**

Artifacts:

- `outputs/geometry/object_mask/best.pt`
- `outputs/geometry/object_mask/last.pt`
- `outputs/geometry/object_mask/history.json`
- `outputs/geometry/object_mask/config.json`


---


### 2.2 Absolute XY Geometry

**Status:** Completed

| Metric | Result |
|---|---:|
| Best validation IoU | **0.3291** |
| Final validation IoU | 0.3218 |
| Final validation Dice | 0.4101 |
| Final training IoU | 0.4475 |
| Final training loss | 0.4932 |
| Final validation loss | 0.6516 |
| Total epochs | 20 |
| Training runtime | ~1 h 29 min |

Artifacts:

- `outputs/geometry/absolute_xy/best.pt`
- `outputs/geometry/absolute_xy/last.pt`
- `outputs/geometry/absolute_xy/history.json`
- `outputs/geometry/absolute_xy/config.json`


---


### 2.3 Object-Relative UV Geometry

**Status:** Completed

| Metric | Result |
|---|---:|
| Best validation IoU | **0.3301** |
| Final validation IoU | 0.3208 |
| Final validation Dice | 0.4082 |
| Final training IoU | 0.4509 |
| Final training loss | 0.4895 |
| Final validation loss | 0.6533 |
| Total epochs | 20 |
| Training runtime | ~1 h 30 min |

Artifacts:

- `outputs/geometry/relative_uv/best.pt`
- `outputs/geometry/relative_uv/last.pt`
- `outputs/geometry/relative_uv/history.json`
- `outputs/geometry/relative_uv/config.json`


---

## 3. Part-Query Alignment Experiments

| Method | Best Val IoU | Val Dice | Status |
|---|---:|---:|---|
| Mask baseline | **0.2977** | 0.3910 | Completed |
| Alignment + mask | **0.3084** | 0.4029 | Completed |
| Alignment + relative UV | **0.3110** | 0.4061 | Completed |

---


### 3.1 Alignment Mask Baseline

**Status:** Completed

| Metric | Result |
|---|---:|
| Best epoch | 13 |
| Best validation IoU | **0.2977** |
| Validation Dice at best epoch | **0.3910** |
| Validation loss at best epoch | 0.6378 |
| Train IoU at best epoch | 0.3529 |
| Train loss at best epoch | 0.5690 |
| Total epochs | 20 |
| Training runtime | ~37 min |

Artifacts:

- `outputs/experiments/part_query_alignment/mask_baseline/best.pt`
- `outputs/experiments/part_query_alignment/mask_baseline/last.pt`
- `outputs/experiments/part_query_alignment/mask_baseline/history.json`
- `outputs/experiments/part_query_alignment/mask_baseline/config.json`


---


### 3.2 Alignment + Mask

**Status:** Completed

| Metric | Result |
|---|---:|
| Best epoch | 13 |
| Best validation IoU | **0.3084** |
| Validation Dice at best epoch | **0.4029** |
| Validation loss at best epoch | 0.7736 |
| Train IoU at best epoch | 0.3693 |
| Train loss at best epoch | 0.6899 |
| Alignment IoU at best epoch | **0.4882** |
| Total epochs | 20 |
| Training runtime | ~35 min |

Artifacts:

- `outputs/experiments/part_query_alignment/alignment_mask/best.pt`
- `outputs/experiments/part_query_alignment/alignment_mask/last.pt`
- `outputs/experiments/part_query_alignment/alignment_mask/history.json`
- `outputs/experiments/part_query_alignment/alignment_mask/config.json`


---


### 3.3 Alignment + Relative UV

**Status:** Completed

| Metric | Result |
|---|---:|
| Best epoch | 13 |
| Best validation IoU | **0.3110** |
| Validation Dice at best epoch | **0.4061** |
| Validation loss at best epoch | 0.7650 |
| Train IoU at best epoch | 0.3833 |
| Train loss at best epoch | 0.6719 |
| Alignment IoU at best epoch | **0.4920** |
| Total epochs | 20 |
| Training runtime | ~37 min |

Artifacts:

- `outputs/experiments/part_query_alignment/alignment_relative_uv/best.pt`
- `outputs/experiments/part_query_alignment/alignment_relative_uv/last.pt`
- `outputs/experiments/part_query_alignment/alignment_relative_uv/history.json`
- `outputs/experiments/part_query_alignment/alignment_relative_uv/config.json`


---

## 4. Object-Centric Zoom Experiments

| Method | Best Val IoU | Val Dice | Status |
|---|---:|---:|---|
| Mask baseline | **0.2977** | 0.3910 | Completed |
| Alignment + mask | **0.3084** | 0.4029 | Completed |
| Alignment + relative UV | **0.3110** | 0.4061 | Completed |

---


### 4.1 Object Zoom Mask Baseline

**Status:** Completed

| Metric | Result |
|---|---:|
| Best epoch | 15 |
| Best validation IoU | **0.3732** |
| Validation Dice at best epoch | **0.4720** |
| Validation loss at best epoch | 0.5909 |
| Train IoU at best epoch | 0.4480 |
| Train loss at best epoch | 0.4977 |
| Total epochs | 20 |
| Training runtime | ~1 h 36 min |

Artifacts:

- `outputs/object_zoom/mask_baseline/best.pt`
- `outputs/object_zoom/mask_baseline/last.pt`
- `outputs/object_zoom/mask_baseline/history.json`
- `outputs/object_zoom/mask_baseline/config.json`


---


### 4.2 Object Zoom + Alignment Mask

**Status:** Completed

| Metric | Result |
|---|---:|
| Best epoch | 13 |
| Best validation IoU | **0.3878** |
| Validation Dice at best epoch | **0.4883** |
| Validation loss at best epoch | 0.7128 |
| Train IoU at best epoch | 0.4552 |
| Train loss at best epoch | 0.6279 |
| Total epochs | 20 |
| Training runtime | ~1 h 38 min |

Artifacts:

- `outputs/object_zoom/alignment_mask/best.pt`
- `outputs/object_zoom/alignment_mask/last.pt`
- `outputs/object_zoom/alignment_mask/history.json`
- `outputs/object_zoom/alignment_mask/config.json`


---


### 4.3 Object Zoom + Alignment Relative UV

**Status:** Completed

| Metric | Result |
|---|---:|
| Best epoch | 13 |
| Best validation IoU | **0.3847** |
| Validation Dice at best epoch | **0.4834** |
| Validation loss at best epoch | 0.7167 |
| Train IoU at best epoch | 0.4545 |
| Train loss at best epoch | 0.6287 |
| Total epochs | 20 |
| Training runtime | ~1 h 38 min |

Artifacts:

- `outputs/object_zoom/alignment_relative_uv/best.pt`
- `outputs/object_zoom/alignment_relative_uv/last.pt`
- `outputs/object_zoom/alignment_relative_uv/history.json`
- `outputs/object_zoom/alignment_relative_uv/config.json`


---

## 5. Final Seen vs Unseen Evaluation

The best validation model family, `alignment_mask`, was evaluated on both `test_seen` and `test_unseen`.

### Final Test Results

| Split | Full-image IoU | Object-crop IoU | Improvement |
|---|---:|---:|---:|
| Test seen | 0.3422 | **0.3766** | +0.0344 |
| Test unseen | 0.2897 | **0.3169** | +0.0272 |

Object-centric zoom improves IoU on both seen and unseen categories.

The final selected object-centric model achieves:

- **Seen IoU:** 0.3766
- **Unseen IoU:** 0.3169

Results file:

- `outputs/object_zoom_evaluation/object_zoom_results.json`

## Notes

- These values come from full training, not smoke tests.
- Test data is not used for model selection.
- `part_only` completed all 20 epochs successfully.

## Query-Gated UVD Geometry

### Fixed UVD

This experiment extends the object-relative geometry representation from relative U/V coordinates to fixed U/V/D geometry, where D is the normalized distance to the parent-object boundary.

| Metric | Result |
|---|---:|
| Best epoch | 16 |
| Best validation IoU | 0.3336 |
| Validation Dice | 0.4245 |
| Validation loss | 0.6325 |
| Train IoU | 0.4176 |
| Train loss | 0.5252 |
| Learning rate | 5.0e-4 |

The fixed UVD representation achieved a validation IoU of **0.3336**, compared with **0.3301** for the earlier relative-UV geometry model. This is an absolute improvement of approximately **0.0035 IoU**, indicating a small benefit from including normalized boundary-distance information.

### Query-Gated UVD

| Metric | Result |
|---|---:|
| Best epoch | 17 |
| Best validation IoU | **0.3273** |
| Validation Dice | **0.4178** |
| Validation loss | 0.6408 |
| Train IoU | 0.4244 |
| Train loss | 0.5172 |
| Learning rate | 5.0e-4 |

The query-gated UVD model achieved a validation IoU of **0.3273**, compared with **0.3336** for fixed UVD and **0.3301** for the earlier relative-UV geometry model.

Therefore, query-conditioned weighting did not improve validation IoU in this full-image geometry experiment. The fixed UVD representation performed best among these three geometry variants.

### UVD Validation Comparison

| Geometry | Best validation IoU |
|---|---:|
| Relative UV | 0.3301 |
| Fixed UVD | **0.3336** |
| Query-Gated UVD | 0.3273 |

The boundary-distance channel provides a small improvement when used directly, while the learned query gate does not improve validation performance in this setting.

### 4.4 Object Zoom + Alignment Fixed UVD

**Status:** Completed

| Metric | Result |
|---|---:|
| Best epoch | 15 |
| Best validation IoU | **0.3829** |
| Validation Dice at best epoch | **0.4830** |
| Validation loss at best epoch | 0.7186 |
| Train IoU at best epoch | 0.4644 |
| Train loss at best epoch | 0.6168 |
| Total epochs | 20 |
| Training runtime | ~1 h 42 min |

Artifacts:

- `outputs/object_zoom/alignment_fixed_uvd/best.pt`
- `outputs/object_zoom/alignment_fixed_uvd/last.pt`
- `outputs/object_zoom/alignment_fixed_uvd/history.json`
- `outputs/object_zoom/alignment_fixed_uvd/config.json`

Compared with the existing crop variants, fixed UVD does not improve validation IoU over `alignment_mask` (0.3878) or `alignment_relative_uv` (0.3847).

### 4.5 Object Zoom + Alignment Query-Gated UVD

**Status:** Completed

| Metric | Result |
|---|---:|
| Best epoch | 15 |
| Best validation IoU | **0.3835** |
| Validation Dice at best epoch | **0.4809** |
| Validation loss at best epoch | 0.7214 |
| Train IoU at best epoch | 0.4685 |
| Train loss at best epoch | 0.6123 |
| Total epochs | 20 |
| Training runtime | ~1 h 41 min |

Artifacts:

- `outputs/object_zoom/alignment_query_gated_uvd/best.pt`
- `outputs/object_zoom/alignment_query_gated_uvd/last.pt`
- `outputs/object_zoom/alignment_query_gated_uvd/history.json`
- `outputs/object_zoom/alignment_query_gated_uvd/config.json`

### Object-Zoom Validation Comparison

| Model | Best validation IoU |
|---|---:|
| Object Zoom | 0.3732 |
| Object Zoom + Alignment Mask | **0.3878** |
| Object Zoom + Alignment Relative UV | 0.3847 |
| Object Zoom + Alignment Fixed UVD | 0.3829 |
| Object Zoom + Alignment Query-Gated UVD | 0.3835 |

The query-gated UVD variant slightly improves over fixed UVD, but neither UVD variant outperforms the simpler alignment-mask model. Therefore, the best object-centric validation model remains `alignment_mask`.

### Final Crop-UVD Seen vs Unseen Evaluation

| Model | Seen IoU | Seen Dice | Unseen IoU | Unseen Dice |
|---|---:|---:|---:|---:|
| Object Zoom + Alignment Mask | **0.3766** | - | **0.3169** | - |
| Object Zoom + Alignment Fixed UVD | 0.3728 | 0.4778 | 0.3091 | 0.4090 |
| Object Zoom + Alignment Query-Gated UVD | 0.3752 | **0.4791** | 0.3074 | 0.4056 |

The query-gated UVD model slightly improves seen-category IoU over fixed UVD (0.3752 vs. 0.3728), but performs slightly worse on unseen categories (0.3074 vs. 0.3091).

Neither UVD variant surpasses the simpler object-centric alignment-mask model, which remains the final selected model with **0.3766 seen IoU** and **0.3169 unseen IoU**.

This suggests that object-centric cropping and query alignment provide the largest benefit in this setup, while the additional U/V/D geometry and query-conditioned gating do not improve generalization to unseen parent-object categories.
