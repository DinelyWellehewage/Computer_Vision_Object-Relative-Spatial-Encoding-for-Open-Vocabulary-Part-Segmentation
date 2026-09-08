import streamlit as st


st.set_page_config(
    page_title="Open-Vocabulary Part Segmentation",
    page_icon="🧩",
    layout="wide",
)


st.title(
    "Text-Conditioned Object-Relative Geometry "
    "for Open-Vocabulary Part Segmentation"
)

st.caption(
    "Dinely Shanuka Welle Hewage · "
    "Sahil Rahul Fulfagar · "
    "Sunil Bharatbhai Talaviya"
)


st.markdown(
    """
### Multimodal AI Systems

This dashboard presents the experiments for text-conditioned
open-vocabulary object-part segmentation.

The system combines:

- frozen **DINOv2** visual features,
- frozen **CLIP** text features,
- a parent-object mask,
- object-relative **U/V/D geometry**,
- query-conditioned geometry weighting,
- object-centric cropping,
- and a lightweight segmentation decoder.

Use the pages in the sidebar to explore the project, compare
experiments, inspect training behavior, study seen-vs-unseen
generalization, and run qualitative predictions.
"""
)


st.divider()


c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Train samples",
    "32,698",
)

c2.metric(
    "Validation samples",
    "3,708",
)

c3.metric(
    "Test seen",
    "3,371",
)

c4.metric(
    "Test unseen",
    "1,586",
)


st.divider()


st.subheader(
    "Research Question"
)

st.info(
    """
Can query-adaptive object-relative geometry improve
open-vocabulary part segmentation by exploiting complementary
horizontal, vertical, and boundary-distance cues, particularly
for unseen parent-object categories?
"""
)


st.subheader(
    "Main Experimental Finding"
)

st.success(
    """
Object-centric cropping combined with query alignment provides
the strongest overall performance.

The simpler Object Zoom + Alignment Mask model achieves the
best validation IoU and the best unseen-category test IoU.

Adding fixed U/V/D geometry or query-gated U/V/D does not
further improve unseen-category generalization.
"""
)


st.subheader(
    "Dashboard"
)

st.markdown(
    """
Use the sidebar to navigate:

**Project Overview**  
Method, motivation, dataset, and experimental design.

**Experiment Results**  
Validation ranking across all completed experiments.

**Training Curves**  
Epoch-by-epoch IoU, Dice, and loss.

**Seen vs Unseen**  
Final generalization comparison.

**Qualitative Demo**  
Run a trained model on an unseen test sample and visualize
its prediction.

**Geometry Analysis**  
Inspect U, V, D maps and query-conditioned geometry weights.
"""
)
