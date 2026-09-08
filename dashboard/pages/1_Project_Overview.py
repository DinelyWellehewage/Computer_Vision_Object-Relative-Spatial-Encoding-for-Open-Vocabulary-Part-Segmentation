import streamlit as st


st.set_page_config(
    page_title="Project Overview",
    page_icon="🧩",
    layout="wide",
)


st.title(
    "Project Overview"
)


st.subheader(
    "Problem"
)

st.write(
    """
Given an RGB image, a parent-object mask, and a natural-language
part query such as **wheel**, **head**, or **wing**, the goal is
to predict the pixel-level mask of the requested object part.
"""
)


st.subheader(
    "Motivation"
)

st.markdown(
    """
Part segmentation is challenging because object parts may be:

- small,
- thin,
- repeated,
- visually similar,
- or dependent on their location inside the parent object.

A parent-object mask identifies which object should be considered,
but does not explicitly encode where pixels lie within that object.
"""
)


st.subheader(
    "Research Question"
)

st.info(
    """
Can query-adaptive object-relative geometry improve
open-vocabulary part segmentation by exploiting complementary
geometric cues — horizontal position, vertical position, and
boundary distance — particularly for unseen parent-object
categories?
"""
)


st.divider()


st.subheader(
    "Architecture"
)

st.markdown(
    r"""
### Visual representation

A frozen **DINOv2 ViT-S/14** encoder extracts dense image features.

### Language representation

A frozen **CLIP ViT-B/32** encoder represents the requested
part query.

### Parent-object conditioning

The provided parent-object mask constrains the segmentation
problem to the selected object.

### Query-conditioned geometry

For the query-gated model, the CLIP text embedding predicts weights:

\[
\alpha_U,\qquad
\alpha_V,\qquad
\alpha_D
\]

which determine the relative importance of each spatial cue.
"""
)



st.divider()


st.subheader(
    "Dataset"
)

c1, c2, c3, c4 = st.columns(
    4
)

c1.metric(
    "Train",
    "32,698",
)

c2.metric(
    "Validation",
    "3,708",
)

c3.metric(
    "Test Seen",
    "3,371",
)

c4.metric(
    "Test Unseen",
    "1,586",
)


st.write(
    """
The experiments use Pascal-Part-116 and explicitly separate
parent-object categories into seen and unseen evaluation groups.
"""
)
st.divider()

st.subheader(
    "Method Pipeline"
)

st.write(
    """
The model combines visual, language, object-mask, and geometric
information before predicting the requested part mask.
"""
)


row1 = st.columns(
    [
        1,
        0.25,
        1,
        0.25,
        1,
    ]
)


with row1[0]:
    st.info(
        """
### RGB Image

Input image containing
the selected object.
"""
    )


with row1[1]:
    st.markdown(
        "## →"
    )


with row1[2]:
    st.info(
        """
### Object-Centric Crop

The parent object is
cropped with surrounding
context.
"""
    )


with row1[3]:
    st.markdown(
        "## →"
    )


with row1[4]:
    st.info(
        """
### DINOv2

Frozen encoder extracts
dense visual features.
"""
    )


st.markdown(
    "### +"
)


row2 = st.columns(
    3
)


with row2[0]:
    st.info(
        """
### CLIP Query

The requested part name
is encoded by frozen CLIP.

Examples:

- head
- wheel
- wing
"""
    )


with row2[1]:
    st.info(
        """
### Parent Mask

Restricts reasoning to
the selected parent
object.
"""
    )


with row2[2]:
    st.info(
        """
### U / V / D Geometry

**U** — horizontal position

**V** — vertical position

**D** — boundary distance
"""
    )


st.markdown(
    "## ↓"
)


row3 = st.columns(
    [
        1,
        0.25,
        1,
        0.25,
        1,
    ]
)


with row3[0]:
    st.success(
        """
### Query Alignment

Measures compatibility
between visual features
and the text query.
"""
    )


with row3[1]:
    st.markdown(
        "## +"
    )


with row3[2]:
    st.success(
        """
### Geometry Gate

For query-gated UVD,
CLIP predicts:

αU, αV, αD
"""
    )


with row3[3]:
    st.markdown(
        "## →"
    )


with row3[4]:
    st.success(
        """
### Decoder

Fuses visual, semantic,
mask, and geometry cues.
"""
    )


st.markdown(
    "## ↓"
)


st.success(
    """
### Predicted Part Mask

The final output is a pixel-level segmentation mask for the
requested object part.
"""
)


st.divider()

st.subheader(
    "Experimental Progression"
)

st.markdown(
    """
The experiments progressively test which information provides
useful spatial guidance:

1. **Part Only**  
   Visual features + text query.

2. **Object Mask**  
   Adds parent-object conditioning.

3. **Absolute XY**  
   Adds image-relative coordinates.

4. **Relative UV**  
   Replaces absolute position with object-relative position.

5. **Fixed UVD**  
   Adds normalized distance from the object boundary.

6. **Query-Gated UVD**  
   Learns query-dependent U/V/D weights.

7. **Object-Centric Crop**  
   Zooms into the selected parent object.

8. **Object-Centric + Alignment + UVD**  
   Combines the strongest semantic and geometric components.
"""
)


st.subheader(
    "Evaluation Metrics"
)

st.markdown(
    """
- Mean Intersection over Union
- Dice score
- Seen-category performance
- Unseen-category performance
- Prediction leakage outside parent object
- Part-size analysis
"""
)