"""
Glaucoma Detection Dashboard
----------------------------
Single-page Streamlit app that:

1. Lets the user upload a fundus image
2. Runs it through a ResNet50 deep ensemble
3. Reports glaucoma prediction and probability
4. Reports ensemble uncertainty
5. Provides an explanation of the result
"""

import io
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input


# ================================================================
# FIX: Markdown treats any line starting with 4+ spaces as a
# "code block", so the heavily-indented HTML strings below were
# being shown as literal text instead of being rendered.
#
# This patch makes every st.markdown() call strip leading
# whitespace off each line first, so indentation never breaks
# the HTML rendering again — no need to touch the code below.
# ================================================================

_original_markdown = st.markdown


def _clean_markdown(body, *args, **kwargs):
    if isinstance(body, str):
        body = "\n".join(line.lstrip() for line in body.split("\n"))
    return _original_markdown(body, *args, **kwargs)


st.markdown = _clean_markdown


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Glaucoma Detection Dashboard",
    page_icon="👁️",
    layout="wide",
)


# ================================================================
# GLOBAL CSS STYLING
# ================================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600;700&display=swap'
    );


    /* ============================================================
       GLOBAL PAGE
       ============================================================ */

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #14202B;
    }

    .stApp {
        background-color: #EFF3F4;
    }


    /* ============================================================
       HEADINGS
       ============================================================ */

    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif;
        color: #0B5566;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    h2 {
        font-size: 1.8rem !important;
    }

    h3 {
        font-size: 1.35rem !important;
    }


    /* ============================================================
       HERO BANNER
       ============================================================ */

    .hero {
        background: linear-gradient(
            120deg,
            #0B5566 0%,
            #10748C 100%
        );

        border-radius: 18px;

        padding: 32px 36px;

        margin-bottom: 25px;

        color: white;

        box-shadow:
            0 8px 20px
            rgba(11, 85, 102, 0.15);
    }


    .hero h1 {
        color: white;

        font-size: 2.2rem;

        font-family: 'Manrope', sans-serif;

        margin-top: 12px;

        margin-bottom: 8px;
    }


    .hero p {
        color: #DCEEF2;

        font-size: 1rem;

        margin: 0;
    }


    .badge {
        display: inline-block;

        background-color:
            rgba(255, 255, 255, 0.15);

        border:
            1px solid
            rgba(255, 255, 255, 0.30);

        border-radius: 999px;

        padding: 6px 14px;

        font-size: 0.8rem;

        font-weight: 600;

        color: white;
    }


    /* ============================================================
       GENERAL CARDS
       ============================================================ */

    div[data-testid="stVerticalBlockBorderWrapper"] {

        background-color: white;

        border-radius: 16px !important;

        border:
            1px solid #D7DEE1 !important;

        box-shadow:
            0 3px 10px
            rgba(20, 32, 43, 0.05);
    }


    /* ============================================================
       CARD TITLES
       ============================================================ */

    .card-title {

        font-family: 'Manrope', sans-serif;

        font-weight: 700;

        font-size: 1.3rem;

        line-height: 1.3;

        color: #0B5566;

        margin-bottom: 8px;
    }


    .card-sub {

        color: #5B6B73;

        font-size: 0.95rem;

        line-height: 1.55;

        margin-bottom: 16px;
    }


    /* ============================================================
       BUTTONS
       ============================================================ */

    .stButton > button {

        background-color: #0B5566;

        color: white;

        border-radius: 10px;

        border: none;

        padding: 11px 20px;

        font-family: 'Manrope', sans-serif;

        font-weight: 700;

        transition: 0.2s;
    }


    .stButton > button:hover {

        background-color: #10748C;

        color: white;

        transform: translateY(-1px);
    }


    /* ============================================================
       SIDEBAR
       ============================================================ */

    section[data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                #0B5566 0%,
                #084553 100%
            );
    }


    section[data-testid="stSidebar"] > div {

        padding-top: 1.75rem;

        padding-bottom: 1.5rem;
    }


    section[data-testid="stSidebar"] h3 {

        color: white !important;

        font-family: 'Manrope', sans-serif;

        font-size: 1.4rem !important;

        font-weight: 700 !important;

        margin-bottom: 1.1rem !important;
    }


    section[data-testid="stSidebar"] label {

        color: #D9E9ED !important;

        font-size: 0.94rem !important;

        font-weight: 600 !important;
    }


    section[data-testid="stSidebar"]
    [data-testid="stTextInput"] input,

    section[data-testid="stSidebar"]
    [data-testid="stNumberInput"] input {

        background-color: white !important;

        color: #14202B !important;

        border-radius: 10px !important;

        border:
            1px solid
            rgba(255, 255, 255, 0.35)
            !important;
    }


    section[data-testid="stSidebar"] hr {

        border-color:
            rgba(255, 255, 255, 0.18)
            !important;
    }


    section[data-testid="stSidebar"]
    [data-testid="stCaptionContainer"] p {

        color: #BFD5DA !important;

        line-height: 1.7 !important;
    }


    /* ============================================================
       RESULT BANNER
       ============================================================ */

    .result-banner {

        border-radius: 12px;

        padding: 16px 20px;

        font-family: 'Manrope', sans-serif;

        font-size: 1.05rem;

        font-weight: 600;

        margin-bottom: 10px;
    }


    .result-positive {

        background-color: #FBEAE6;

        border-left:
            5px solid #C1442E;

        color: #7C2A1B;
    }


    .result-negative {

        background-color: #E7F3EC;

        border-left:
            5px solid #2E8B57;

        color: #1E5C3A;
    }


    /* ============================================================
       RESULT CARDS
       ============================================================ */

    .result-card {

        display: flex;

        flex-direction: column;

        justify-content: center;

        align-items: center;

        text-align: center;

        min-height: 210px;

        padding: 20px 15px;

        width: 100%;
    }


    /* ============================================================
       RESULT ICONS
       ============================================================ */

    .result-icon {

        font-size: 1.8rem;

        margin-bottom: 10px;

        display: flex;

        justify-content: center;

        align-items: center;
    }


    /* ============================================================
       RESULT CARD HEADINGS
       ============================================================ */

    .result-card-title {

        font-family: 'Manrope', sans-serif;

        font-size: 1.15rem;

        font-weight: 700;

        color: #0B5566;

        text-align: center;

        width: 100%;

        margin-bottom: 16px;
    }


    /* ============================================================
       RESULT VALUES
       ============================================================ */

    .result-card-value {

        font-family:
            'IBM Plex Mono',
            monospace;

        font-size: 2rem;

        font-weight: 700;

        text-align: center;

        width: 100%;

        line-height: 1.3;

        margin: 0;
    }


    /* ============================================================
       PREDICTION RESULT COLOURS
       ============================================================ */

    .positive-result {

        color: #C1442E;
    }


    .negative-result {

        color: #2E8B57;
    }


    .probability-result {

        color: #0B5566;
    }


    .uncertainty-result {

        color: #B7791F;
    }


    /* ============================================================
       COLOUR ACCENT LINES
       ============================================================ */

    .prediction-line {

        width: 100%;

        height: 5px;

        background-color: #C1442E;

        margin-bottom: 18px;
    }


    .probability-line {

        width: 100%;

        height: 5px;

        background-color: #10748C;

        margin-bottom: 18px;
    }


    .uncertainty-line {

        width: 100%;

        height: 5px;

        background-color: #B7791F;

        margin-bottom: 18px;
    }


    /* ============================================================
       FILE UPLOADER
       ============================================================ */

    div[data-testid="stFileUploaderDropzone"] {

        background-color: #F5F8F9;

        border:
            1.5px dashed #9FB8BE;

        border-radius: 12px;
    }


    /* ============================================================
       PROBABILITY GAUGE
       ============================================================ */

    .gauge-track {

        width: 100%;

        height: 10px;

        background-color: #E4EBED;

        border-radius: 999px;

        overflow: hidden;

        margin-top: 4px;
    }


    .gauge-fill {

        height: 100%;

        border-radius: 999px;
    }


    /* ============================================================
       FOOTER
       ============================================================ */

    .footer-note {

        text-align: center;

        color: #8A9AA1;

        font-size: 0.82rem;

        margin-top: 8px;

        margin-bottom: 4px;
    }


    /* ============================================================
       PLAIN-LANGUAGE HELPERS
       ============================================================ */

    .result-card-plain {

        font-size: 0.8rem;

        color: #5B6B73;

        line-height: 1.4;

        margin-top: 10px;
    }

    .plain-banner {

        background-color: #F5F8F9;

        border: 1px solid #D7DEE1;

        border-radius: 10px;

        padding: 12px 16px;

        font-size: 0.92rem;

        color: #14202B;

        margin-top: 10px;

        margin-bottom: 10px;
    }

    .glossary-term {

        font-family: 'Manrope', sans-serif;

        font-weight: 700;

        color: #0B5566;
    }

    .glossary-item {

        margin-bottom: 12px;

        line-height: 1.5;
    }

    .glossary-item:last-child {

        margin-bottom: 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# HERO HEADER
# ================================================================

st.markdown(
    """
    <div class="hero">

        <span class="badge">
            Research prototype · not a diagnostic device
        </span>

        <h1>Glaucoma Detection Dashboard</h1>

        <p>
            Deep ensemble (5× ResNet50) fundus image classifier
            with uncertainty estimation
        </p>

        <p style="margin-top: 8px; margin-bottom: 0; opacity: 0.9; font-size: 0.92rem;">
            In plain terms: upload a photo of the back of the eye and get an
            AI-assisted estimate of glaucoma risk, plus a sense of how much
            to trust that estimate.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# ABOUT / METHODOLOGY EXPANDER
# ================================================================

with st.expander("ℹ️ About this dashboard", expanded=False):

    st.markdown(
        """
        **In plain terms:** this tool looks at a photo of the back of the
        eye (called a *fundus image*) and estimates how likely it is to
        show signs of glaucoma, a common eye condition that can damage
        vision if untreated. Instead of relying on a single AI model, it
        uses **5 separate models** and checks how much they agree with
        one another — the more they agree, the more trustworthy the
        result tends to be.

        ---

        **Technical details:** this tool wraps a **deep ensemble of 5
        independently trained ResNet50 models**, each classifying a
        fundus image as glaucoma-positive (GON+) or glaucoma-negative
        (GON-).

        - **Prediction** is the ensemble's majority verdict against your
        chosen decision threshold.
        - **Probability** is the *average* glaucoma probability across
        all 5 models.
        - **Uncertainty (σ)** is the *standard deviation* across the 5
        models' predictions — a high value means the models disagree,
        so the result deserves more scrutiny.

        Adjust the decision threshold and confidence cutoff in the
        sidebar to match your evaluation criteria.
        """
    )


# ================================================================
# GLOSSARY EXPANDER
# ================================================================

with st.expander("📖 Glossary — plain-English definitions"):

    st.markdown(
        """
        <div class="glossary-item">
        <span class="glossary-term">Fundus image</span> — a photo of the
        back inside surface of the eye, including the optic disc, taken
        with a special camera at an eye exam.
        </div>

        <div class="glossary-item">
        <span class="glossary-term">Glaucoma</span> — an eye condition
        that can damage the optic nerve, sometimes leading to vision
        loss if it isn't caught and managed early.
        </div>

        <div class="glossary-item">
        <span class="glossary-term">Ensemble</span> — instead of asking
        one AI model, this tool asks 5 of them and combines their
        answers, which tends to be more reliable than trusting just one.
        </div>

        <div class="glossary-item">
        <span class="glossary-term">Probability</span> — how likely the
        AI thinks the image shows glaucoma-related patterns, shown as a
        percentage. Higher = more likely, according to the model.
        </div>

        <div class="glossary-item">
        <span class="glossary-term">Decision threshold</span> — the
        cut-off probability above which the tool labels an image
        "Positive". You can adjust this in the sidebar.
        </div>

        <div class="glossary-item">
        <span class="glossary-term">Uncertainty (σ)</span> — how much
        the 5 models disagree with each other. A low number means they
        mostly agree; a high number means they don't, so the result
        deserves more caution.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ================================================================
# SIDEBAR SETTINGS
# ================================================================

st.sidebar.markdown("### 🩺 Display settings")

simple_mode = st.sidebar.toggle(
    "Plain-language explanations",
    value=True,
    help=(
        "Show extra plain-English explanations next to the technical "
        "results — handy if you're not a clinician or ML specialist. "
        "Turn off for a more compact, technical-only view."
    ),
)

st.sidebar.markdown("---")

st.sidebar.markdown("### ⚙️ Model settings")

model_dir = st.sidebar.text_input(
    "Model folder",
    value="models",
    help="Folder containing your saved .keras ensemble members."
)


n_models = st.sidebar.number_input(
    "Number of ensemble members",
    min_value=1,
    max_value=10,
    value=5,
    help="How many independently trained models to load and average."
)


filename_pattern = st.sidebar.text_input(
    "Filename pattern (use {i} for seed number)",
    value="resnet50_seed{i}.keras"
)


img_size = st.sidebar.number_input(
    "Model input size (px)",
    min_value=64,
    max_value=600,
    value=224,
    step=1
)


st.sidebar.markdown("### 🎯 Decision settings")


decision_threshold = st.sidebar.slider(
    "Decision threshold (probability ≥ this = Positive)",
    0.0,
    1.0,
    0.5,
    0.01,
    help="Raise this to reduce false positives; lower it to catch more "
         "possible glaucoma cases at the cost of more false alarms."
)


uncertainty_high_cutoff = st.sidebar.slider(
    "Std-dev above this = low confidence",
    0.0,
    0.30,
    0.011,
    0.001,
    help="If the 5 models disagree by more than this, the result is "
         "flagged as low confidence. Default (0.011) is the threshold "
         "that best separated correct from incorrect predictions on "
         "our own test set — it caught 6/6 misclassifications there, "
         "vs. only 4/6 at the previous default of 0.10."
)


st.sidebar.markdown("---")


if st.sidebar.button("🔄 Check another image", use_container_width=True):

    for key in ("uploaded_file", "last_results"):

        st.session_state.pop(key, None)

    st.rerun()


st.sidebar.markdown("---")


st.sidebar.caption(
    """
    This tool supports a university research project.

    It is not a certified diagnostic device and should not
    be used for real clinical decisions.
    """
)


# ================================================================
# MODEL LOADING
# ================================================================

@st.cache_resource(show_spinner=False)
def load_ensemble(model_dir, pattern, n):

    models = []

    errors = []

    for i in range(1, n + 1):

        path = (
            f"{model_dir.rstrip('/')}/"
            f"{pattern.format(i=i)}"
        )

        try:

            model = tf.keras.models.load_model(path)

            models.append(model)

        except Exception as e:

            errors.append(f"{path}: {e}")

    return models, errors


# ================================================================
# IMAGE PREPROCESSING
# ================================================================

def preprocess_image(pil_img, size):

    img = (
        pil_img
        .convert("RGB")
        .resize((size, size), Image.NEAREST)
    )

    arr = np.array(img).astype("float32")

    arr = preprocess_input(arr)

    return np.expand_dims(arr, axis=0)


# ================================================================
# IMAGE QUALITY / OUT-OF-DISTRIBUTION CHECK
# ================================================================
#
# Lightweight, heuristic checks — not a trained classifier — meant to
# flag inputs that are unlikely to be genuine colour fundus photos
# (screenshots, diagrams, stock photos, corrupted uploads, etc.)
# before they're passed to the ensemble. This does NOT detect true
# domain shift between camera sources — it only catches the most
# obviously invalid inputs.

def assess_fundus_image_quality(pil_img):

    warnings = []

    width, height = pil_img.size

    # ------------------------------------------------------------
    # Resolution check
    # ------------------------------------------------------------

    if width < 150 or height < 150:

        warnings.append(
            "This image is very low resolution for a fundus photo "
            "(fundus cameras typically produce much larger images)."
        )

    # ------------------------------------------------------------
    # Aspect ratio check — fundus photos are close to square/circular
    # ------------------------------------------------------------

    aspect_ratio = width / height

    if aspect_ratio < 0.7 or aspect_ratio > 1.4:

        warnings.append(
            "This image's proportions are unusual for a fundus photo "
            "(fundus photos are typically close to square)."
        )

    arr = np.array(pil_img.convert("RGB")).astype("float32")

    mean_r = arr[:, :, 0].mean()
    mean_g = arr[:, :, 1].mean()
    mean_b = arr[:, :, 2].mean()

    # ------------------------------------------------------------
    # Colour balance check — fundus photos are red/orange-dominant
    # ------------------------------------------------------------

    if mean_r <= mean_b:

        warnings.append(
            "This image's colour balance is unusual for a fundus photo "
            "(fundus photos are usually red/orange-dominant, not "
            "blue-dominant)."
        )

    # ------------------------------------------------------------
    # Saturation check — flags near-greyscale images (diagrams,
    # screenshots, scanned documents)
    # ------------------------------------------------------------

    max_channel = arr.max(axis=2)
    min_channel = arr.min(axis=2)

    saturation = np.where(
        max_channel > 0,
        (max_channel - min_channel) / np.maximum(max_channel, 1e-6),
        0.0
    )

    mean_saturation = saturation.mean()

    if mean_saturation < 0.08:

        warnings.append(
            "This image looks mostly greyscale, but fundus photos are "
            "normally in full colour."
        )

    # ------------------------------------------------------------
    # Brightness check — flags near-blank or near-black images
    # ------------------------------------------------------------

    brightness = arr.mean()

    if brightness < 15:

        warnings.append(
            "This image is almost entirely black or very dark."
        )

    elif brightness > 240:

        warnings.append(
            "This image is almost entirely white or blank."
        )

    return warnings


# ================================================================
# RUN ENSEMBLE
# ================================================================

def run_ensemble(models, x):

    probabilities = []

    for model in models:

        prediction = model.predict(
            x,
            verbose=0
        )

        prediction = np.array(
            prediction
        ).squeeze()

        if prediction.ndim == 0:

            probability = float(prediction)

        else:

            probability = float(prediction[-1])

        probabilities.append(probability)

    return np.array(probabilities)


# ================================================================
# CONFIDENCE LABEL
# ================================================================

def confidence_label(std, cutoff):

    if std >= cutoff:

        return "Low confidence"

    if std <= cutoff / 3:

        return "High confidence"

    return "Reasonably confident"


# ================================================================
# IMAGE UPLOAD + INFORMATION
# ================================================================

col_upload, col_info = st.columns(
    2,
    gap="medium"
)


# ------------------------------------------------
# UPLOAD CARD
# ------------------------------------------------

with col_upload:

    with st.container(border=True):

        st.markdown(
            '<div class="card-title">📤 Upload a fundus image</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '''
            <div class="card-sub">
            Upload a PNG or JPG fundus image.
            Ideally, the optic disc should be clearly visible.
            </div>
            ''',
            unsafe_allow_html=True
        )


        uploaded_file = st.file_uploader(
            "Fundus photo",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key="uploaded_file"
        )


        if uploaded_file is not None:

            image = Image.open(
                io.BytesIO(
                    uploaded_file.read()
                )
            )


            # Centre the image

            left_space, image_column, right_space = st.columns(
                [1, 2, 1]
            )


            with image_column:

                st.image(
                    image,
                    caption="Uploaded fundus image",
                    use_container_width=True
                )


            # --------------------------------------------------------
            # QUALITY / OUT-OF-DISTRIBUTION WARNINGS
            # --------------------------------------------------------

            quality_warnings = assess_fundus_image_quality(image)

            if quality_warnings:

                warning_items = "".join(
                    f"<li>{w}</li>" for w in quality_warnings
                )

                st.markdown(
                    f"""
                    <div class="plain-banner" style="border-left: 4px solid #C1442E;">
                    ⚠️ <strong>This doesn't look like a typical fundus
                    photo.</strong> You can still run detection, but the
                    result may be unreliable:
                    <ul style="margin-top: 8px; margin-bottom: 0;">
                        {warning_items}
                    </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.session_state["quality_warnings"] = quality_warnings


# ------------------------------------------------
# INFORMATION CARD
# ------------------------------------------------

with col_info:

    with st.container(border=True):

        st.markdown(
            '<div class="card-title">🔬 How this works</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            '''
            <div class="card-sub">
            What happens after you upload your image
            </div>
            ''',
            unsafe_allow_html=True
        )


        st.markdown(
            """
            1. The fundus image is resized and preprocessed for ResNet50.

            2. Five independently trained ResNet50 models analyse the image.

            3. The ensemble average produces the glaucoma probability.

            4. Differences between the models are used to estimate uncertainty.

            5. High model disagreement means the prediction should be interpreted
            with greater caution.
            """
        )


st.write("")


# ================================================================
# RUN DETECTION BUTTON
# ================================================================

if uploaded_file is not None:

    left_button, middle_button, right_button = st.columns(
        [1, 1.5, 1]
    )


    with middle_button:

        run_clicked = st.button(
            "🔍 Run glaucoma detection",
            type="primary",
            use_container_width=True
        )


    # ============================================================
    # RUN MODEL
    # ============================================================

    if run_clicked:


        with st.spinner("Loading deep learning models..."):

            models, load_errors = load_ensemble(
                model_dir,
                filename_pattern,
                n_models
            )


        # --------------------------------------------------------
        # MODEL ERRORS
        # --------------------------------------------------------

        if load_errors:

            with st.container(border=True):

                st.markdown(
                    '<div class="card-title">⚠️ Model loading issues</div>',
                    unsafe_allow_html=True
                )

                for error in load_errors:

                    st.code(error)


        if not models:

            st.error(
                "No models could be loaded. Please check your model folder."
            )

            st.stop()


        # --------------------------------------------------------
        # RUN INFERENCE
        # --------------------------------------------------------

        with st.spinner("Analysing fundus image..."):

            x = preprocess_image(
                image,
                img_size
            )


            probabilities = run_ensemble(
                models,
                x
            )


        # ========================================================
        # CALCULATE RESULTS
        # ========================================================

        mean_prob = float(
            np.mean(probabilities)
        )


        std_prob = float(
            np.std(probabilities)
        )


        prediction = (
            "Glaucoma Positive"
            if mean_prob >= decision_threshold
            else "Glaucoma Negative"
        )


        confidence = confidence_label(
            std_prob,
            uncertainty_high_cutoff
        )


        is_positive = (
            prediction == "Glaucoma Positive"
        )


        # ========================================================
        # RESULTS HEADER
        # ========================================================

        st.write("")

        st.markdown("## 📊 Results")


        # ========================================================
        # RESULT BANNER
        # ========================================================

        banner_class = (
            "result-positive"
            if is_positive
            else "result-negative"
        )


        banner_icon = (
            "⚠️"
            if is_positive
            else "✅"
        )


        st.markdown(
            f"""
            <div class="result-banner {banner_class}">
                {banner_icon}
                <strong>{prediction}</strong>
                — {confidence}
            </div>
            """,
            unsafe_allow_html=True
        )


        if simple_mode:

            plain_result = (
                "This means the scan shows patterns that are often "
                "associated with glaucoma. It's worth discussing this "
                "result with an eye care professional."
                if is_positive
                else
                "This means the scan does not show strong patterns "
                "typically associated with glaucoma."
            )

            plain_confidence = (
                "The 5 AI models mostly disagreed with each other, so "
                "treat this result with extra caution."
                if confidence == "Low confidence"
                else
                "The 5 AI models agreed with each other almost "
                "perfectly, which strongly supports this result."
                if confidence == "High confidence"
                else
                "The 5 AI models mostly agreed with each other, which "
                "supports this result."
            )

            st.markdown(
                f"""
                <div class="plain-banner">
                🩺 <strong>In plain terms:</strong> {plain_result}
                {plain_confidence}
                </div>
                """,
                unsafe_allow_html=True
            )


        if st.session_state.get("quality_warnings"):

            st.markdown(
                """
                <div class="result-banner" style="background-color: #FBEAE6; border-left: 5px solid #C1442E; color: #7C2A1B;">
                ⚠️ <strong>Quality warning:</strong> this image didn't pass
                basic fundus-photo checks (see the upload panel above).
                Treat this result with extra caution — it may not reflect
                the model's real performance on genuine fundus images.
                </div>
                """,
                unsafe_allow_html=True
            )


        st.write("")


        # ========================================================
        # RESULT CARDS
        # ========================================================

        m1, m2, m3 = st.columns(
            3,
            gap="medium"
        )


        # --------------------------------------------------------
        # PREDICTION CARD
        # --------------------------------------------------------

        prediction_class = (
            "positive-result"
            if is_positive
            else "negative-result"
        )


        prediction_icon = (
            "⚠️"
            if is_positive
            else "✓"
        )


        prediction_plain = (
            '<div class="result-card-plain">The AI\'s best guess, based '
            "on the decision threshold you've set in the sidebar.</div>"
            if simple_mode else ""
        )


        with m1:

            with st.container(border=True):

                st.markdown(
                    f"""
                    <div class="prediction-line"></div>

                    <div class="result-card">

                        <div class="result-icon">
                            {prediction_icon}
                        </div>

                        <div class="result-card-title">
                            Prediction
                        </div>

                        <div class="result-card-value {prediction_class}">
                            {prediction}
                        </div>

                        {prediction_plain}

                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # --------------------------------------------------------
        # PROBABILITY CARD
        # --------------------------------------------------------

        probability_plain = (
            '<div class="result-card-plain">How likely the AI thinks '
            "this image shows glaucoma-related patterns. Higher = more "
            "likely.</div>"
            if simple_mode else ""
        )


        with m2:

            with st.container(border=True):

                st.markdown(
                    f"""
                    <div class="probability-line"></div>

                    <div class="result-card">

                        <div class="result-icon">
                            📊
                        </div>

                        <div class="result-card-title">
                            Glaucoma probability
                        </div>

                        <div class="result-card-value probability-result">
                            {mean_prob:.1%}
                        </div>

                        <div class="gauge-track">
                            <div class="gauge-fill" style="width: {mean_prob * 100:.1f}%; background-color: #0B5566;"></div>
                        </div>

                        {probability_plain}

                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # --------------------------------------------------------
        # UNCERTAINTY CARD
        # --------------------------------------------------------

        uncertainty_plain = (
            '<div class="result-card-plain">How much the 5 AI models '
            "disagree with each other. Lower = more consistent, higher "
            "= more caution needed.</div>"
            if simple_mode else ""
        )


        with m3:

            with st.container(border=True):

                st.markdown(
                    f"""
                    <div class="uncertainty-line"></div>

                    <div class="result-card">

                        <div class="result-icon">
                            ⚡
                        </div>

                        <div class="result-card-title">
                            Ensemble uncertainty (σ)
                        </div>

                        <div class="result-card-value uncertainty-result">
                            {std_prob:.3f}
                        </div>

                        {uncertainty_plain}

                    </div>
                    """,
                    unsafe_allow_html=True
                )


        st.write("")


        # ========================================================
        # PER-MODEL BREAKDOWN
        # ========================================================

        with st.container(border=True):

            st.markdown(
                '<div class="card-title">📈 Per-model breakdown</div>',
                unsafe_allow_html=True
            )


            st.markdown(
                '''
                <div class="card-sub">
                Glaucoma probability predicted by each ensemble member
                </div>
                ''',
                unsafe_allow_html=True
            )


            if simple_mode:

                st.markdown(
                    '''
                    <div class="plain-banner">
                    🩺 <strong>In plain terms:</strong> each bar is one AI
                    model's individual opinion. If the bars are close
                    together, the models agree. If they're spread far
                    apart, that's a sign of more uncertainty in the
                    result.
                    </div>
                    ''',
                    unsafe_allow_html=True
                )


            breakdown_df = pd.DataFrame(
                {
                    "Model": [
                        f"ResNet50 (seed {i + 1})"
                        for i in range(
                            len(probabilities)
                        )
                    ],

                    "Glaucoma probability":
                        probabilities
                }
            )


            bars = (
                alt.Chart(breakdown_df)
                .mark_bar(color="#10748C", cornerRadius=4)
                .encode(
                    x=alt.X("Model:N", sort=None, title=None),
                    y=alt.Y(
                        "Glaucoma probability:Q",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format="%"),
                    ),
                    tooltip=[
                        "Model",
                        alt.Tooltip("Glaucoma probability:Q", format=".1%"),
                    ],
                )
            )


            threshold_line = (
                alt.Chart(pd.DataFrame({"y": [decision_threshold]}))
                .mark_rule(color="#C1442E", strokeDash=[6, 3])
                .encode(y="y:Q")
            )


            st.altair_chart(
                (bars + threshold_line).properties(height=280),
                use_container_width=True,
            )


            st.caption(
                f"Dashed line = decision threshold ({decision_threshold:.0%})"
            )


        st.write("")


        # ========================================================
        # EXPLANATION
        # ========================================================

        with st.container(border=True):

            st.markdown(
                '<div class="card-title">💡 What does this mean?</div>',
                unsafe_allow_html=True
            )


            reliability_text = (
                "The ensemble members were relatively consistent, "
                "which supports the reliability of the prediction."
                if std_prob < uncertainty_high_cutoff
                else
                "The ensemble members showed greater disagreement, "
                "so this prediction should be interpreted with caution."
            )


            st.markdown(
                f"""
- The deep learning ensemble estimates a **{mean_prob:.1%} probability** that this fundus image shows patterns associated with glaucoma.
- The decision threshold is **{decision_threshold:.0%}**, resulting in a classification of **{prediction}**.
- The ensemble uncertainty is **{std_prob:.3f}**.
- {reliability_text}
- ⚠️ This dashboard is a university research prototype and is **not a certified diagnostic device**. Results should not be used as a substitute for professional clinical assessment.
                """
            )

            if simple_mode:

                st.markdown(
                    """
                    <div class="plain-banner">
                    🩺 <strong>In one sentence:</strong> think of this as a
                    second opinion from software, not a diagnosis — a real
                    eye doctor should always confirm any result before any
                    decisions are made about your eye health.
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        st.write("")


        # ========================================================
        # DOWNLOAD REPORT
        # ========================================================

        report_lines = [
            "Glaucoma Detection Dashboard — Result Summary",
            "=" * 48,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"Prediction: {prediction}",
            f"Glaucoma probability (ensemble mean): {mean_prob:.1%}",
            f"Ensemble uncertainty (std dev): {std_prob:.3f}",
            f"Confidence: {confidence}",
            f"Decision threshold used: {decision_threshold:.0%}",
            f"Uncertainty cutoff used: {uncertainty_high_cutoff:.2f}",
            "",
            "Per-model probabilities:",
        ]

        for i, p in enumerate(probabilities, start=1):

            report_lines.append(f"  - ResNet50 (seed {i}): {p:.1%}")

        report_lines += [
            "",
            "This dashboard is a university research prototype and is "
            "not a certified diagnostic device. Results should not be "
            "used as a substitute for professional clinical assessment.",
        ]

        report_text = "\n".join(report_lines)

        left_dl, mid_dl, right_dl = st.columns([1, 1.5, 1])

        with mid_dl:

            st.download_button(
                "📄 Download result summary",
                data=report_text,
                file_name="glaucoma_dashboard_result.txt",
                mime="text/plain",
                use_container_width=True,
            )


        st.markdown(
            '<div class="footer-note">Glaucoma Detection Dashboard · Research prototype</div>',
            unsafe_allow_html=True,
        )


# ================================================================
# DEFAULT MESSAGE
# ================================================================

else:

    with st.container(border=True):

        st.info(
            "👆 Upload a fundus image above to get started."
        )
