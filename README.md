# Text-Conditioned Object-Relative Geometry for Open-Vocabulary Part Segmentation

This project investigates **text-conditioned part segmentation** using frozen DINOv2 visual features, CLIP text embeddings, parent-object masks, and explicit object-relative geometry.

Given an **RGB image**, a **parent-object mask**, and a **text query** such as `"head"`, `"wheel"`, or `"wing"`, the model predicts the segmentation mask of the requested object part.

## Method

The experiments progressively compare:

* Part-only segmentation
* Parent-object mask conditioning
* Absolute image coordinates \(X,Y\)
* Object-relative coordinates \(U,V\)
* Fixed \(U,V,D\) geometry
* Query-gated \(U,V,D\) geometry
* Object-centric cropping
* Query alignment with object-centric geometry

Here, \(D\) represents normalized distance from the parent-object boundary.

The strongest validation model is the **object-centric alignment model**, achieving a validation IoU of **0.3878**. Explicit UVD geometry provides limited additional benefit once object-centric cropping and query alignment are used.

## Dataset

The project uses **Pascal-Part-116** with separate seen and unseen object-category evaluation.

| Split           | Part-query samples |
| --------------- | -----------------: |
| Train Seen      |             32,698 |
| Validation Seen |              3,708 |
| Test Seen       |              3,371 |
| Test Unseen     |              1,586 |

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prepare the dataset:

```bash
python scripts/prepare_dataset.py
```

Inspect the prepared dataset:

```bash
python scripts/inspect_dataset.py --split train
```

## Running Experiments

Baseline experiments:

```bash
python scripts/train_baseline.py --help
```

Geometry experiments:

```bash
python scripts/train_geometry.py --help
```

Query-gated UVD experiments:

```bash
python scripts/train_uvd.py --help
```

Object-centric experiments:

```bash
python scripts/train_object_zoom.py --help
```

Robustness evaluation:

```bash
python scripts/evaluate_robustness.py
```

For full training, the corresponding Slurm scripts under `scripts/` can be submitted on a GPU cluster.

## Dashboard

The project includes an interactive **Streamlit dashboard** for exploring:

* Project methodology
* Experiment results
* Training curves
* Seen vs. unseen performance
* Qualitative segmentation predictions
* U/V/D geometry maps
* Query-conditioned geometry weights

Run the dashboard from the repository root:

```bash
source .venv/bin/activate
streamlit run dashboard/app.py
```

Then open:

```text
http://localhost:8501
```

The qualitative and geometry pages require the corresponding trained checkpoints under `outputs/`.

## Project Structure

```text
datasets/       Dataset loaders and preprocessing
src/            Models, geometry, metrics, and utilities
scripts/        Training and evaluation scripts
notebooks/      Experimental notebooks
dashboard/      Interactive Streamlit dashboard
outputs/        Trained checkpoints and experiment results
```

## Main Finding

Object-relative geometry provides modest spatial guidance, but **object-centric cropping and query alignment provide the largest performance improvement**. Query-conditioned UVD gating does not outperform the simpler alignment model and does not improve unseen-category generalization.
