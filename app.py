import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Sepsis Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

DOCTOR_PATIENT_IMAGE = "doctor_patient.png"

# =========================================================
# LOAD MODEL FILES
# =========================================================

try:
    model = joblib.load("rf_model.pkl")
    imputer = joblib.load("imputer.pkl")
    feature_names = joblib.load("feature_names.pkl")
except Exception as e:
    st.error("Could not load the trained model files.")
    st.code(str(e))
    st.stop()

feature_names = list(feature_names)

# =========================================================
# SELECTED FEATURES
# =========================================================

selected_features = [
    "Hour",
    "Heart_rate__in_beats_per_minute",
    "Respiration_rate__breaths_per_minute",
    "Temperature__deg_C",
    "Mean_arterial_pressure__mm_Hg",
    "Fraction_of_inspired_oxygen__percentage",
    "Systolic_BP__mm_Hg",
    "Age__years",
    "Diastolic_BP__mm_Hg",
    "pH",
    "Partial_pressure_of_carbon_dioxide_from_arterial_blood__mm_Hg",
    "Excess_bicarbonate__mmol_L",
    "Pulse_oximetry__percentage",
    "Administrative_identifier_for_ICU_unit__MICU___false__0__or_true"
]

# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "section" not in st.session_state:
    st.session_state.section = "Patient Data"

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# DEFAULT VALUES
# =========================================================

if "age" not in st.session_state:
    st.session_state.age = 45

if "hour" not in st.session_state:
    st.session_state.hour = 0

if "heart_rate" not in st.session_state:
    st.session_state.heart_rate = 78

if "pulse_ox" not in st.session_state:
    st.session_state.pulse_ox = 98

if "temperature" not in st.session_state:
    st.session_state.temperature = 36.8

if "systolic" not in st.session_state:
    st.session_state.systolic = 120

if "diastolic" not in st.session_state:
    st.session_state.diastolic = 75

if "map_value" not in st.session_state:
    st.session_state.map_value = 80

if "respiration" not in st.session_state:
    st.session_state.respiration = 16

if "fio2" not in st.session_state:
    st.session_state.fio2 = 21

if "paco2" not in st.session_state:
    st.session_state.paco2 = 40

if "bicarbonate" not in st.session_state:
    st.session_state.bicarbonate = 0.0

if "ph" not in st.session_state:
    st.session_state.ph = 7.40

if "icu_unit" not in st.session_state:
    st.session_state.icu_unit = "None"

# =========================================================
# CSS
# =========================================================

st.html("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 90% 5%,
            rgba(99,190,238,.12),
            transparent 25%
        ),
        radial-gradient(
            circle at 5% 80%,
            rgba(117,205,239,.10),
            transparent 25%
        ),
        #f7fbff;
    color: #173f63;
}

.block-container {
    max-width: 1550px;
    padding: 18px 22px 35px;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #d9eaf5;
    min-width: 115px;
    max-width: 115px;
}

section[data-testid="stSidebar"] > div {
    padding: 18px 9px;
}

.nav-logo {
    width: 54px;
    height: 54px;
    margin: 0 auto 24px;
    border-radius: 17px;
    background: #eef8ff;
    border: 1px solid #d8ebf7;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 27px;
    color: #0879d1;
}

.nav-button {
    margin: 7px 0;
}

.nav-button button {
    height: 64px !important;
    padding: 5px !important;
    border-radius: 16px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    color: #668aa5 !important;
    font-size: 10px !important;
    font-weight: 650 !important;
    box-shadow: none !important;
}

.nav-button button:hover {
    background: #eef8ff !important;
    color: #0879d1 !important;
    border-color: #d9eaf5 !important;
}

.nav-active button {
    background: #e7f5ff !important;
    color: #0879d1 !important;
    font-weight: 850 !important;
    border-color: #cfe7f7 !important;
}

.nav-footer {
    margin-top: 35px;
    text-align: center;
    font-size: 7.5px;
    line-height: 1.55;
    color: #7b9eb8;
}

/* HEADER */

.header {
    height: 78px;
    background: #ffffff;
    border: 1px solid #d9eaf5;
    border-radius: 22px;
    padding: 0 23px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 10px 30px rgba(28,113,167,.07);
    position: relative;
    overflow: hidden;
}

.header-title {
    font-size: 27px;
    font-weight: 900;
    color: #073f7c;
}

.header-sub {
    font-size: 12px;
    color: #7094af;
    margin-top: 5px;
}

.header-right {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #527f9f;
    font-size: 11px;
}

.avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: #eaf6ff;
    border: 2px solid white;
    box-shadow: 0 2px 10px #cfe5f4;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #1677bd;
    font-weight: 900;
}

/* GENERAL CARDS */

.panel,
.home-card,
.analysis-card,
.about-box {
    background: #ffffff;
    border: 1px solid #d9eaf5;
    border-radius: 22px;
    box-shadow: 0 12px 34px rgba(32,117,169,.07);
}

.home-card {
    padding: 30px;
}

.home-title {
    font-size: 32px;
    font-weight: 900;
    color: #073f7c;
}

.home-text {
    font-size: 13px;
    color: #6688a5;
    line-height: 1.7;
    max-width: 800px;
}

.home-feature {
    padding: 18px;
    border: 1px solid #d9eaf5;
    border-radius: 16px;
    background: #ffffff;
    height: 130px;
    box-shadow: 0 7px 20px rgba(32,117,169,.05);
}

.home-feature-title {
    font-weight: 900;
    color: #07549a;
    font-size: 13px;
}

.home-feature-text {
    font-size: 10px;
    color: #7595ae;
    line-height: 1.5;
    margin-top: 6px;
}

/* CLINICAL DATA */

.clinical-top {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 16px 18px;
}

.clinical-icon {
    width: 43px;
    height: 43px;
    border-radius: 14px;
    background: #eaf6ff;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #0879d1;
    font-size: 22px;
}

.clinical-title {
    font-size: 21px;
    font-weight: 900;
    color: #07549a;
}

.clinical-sub {
    font-size: 10px;
    color: #7696b0;
    margin-top: 2px;
}

/* STEPS */

.steps {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 7px;
    margin-bottom: 11px;
}

.step-button button {
    height: 61px !important;
    padding: 8px !important;
    border: 1px solid #d9eaf5 !important;
    border-radius: 14px !important;
    background: #ffffff !important;
    color: #07539a !important;
    box-shadow: none !important;
    font-size: 10px !important;
    font-weight: 800 !important;
}

.step-button button:hover {
    background: #eef8ff !important;
    border-color: #c7e4f7 !important;
}

.step-active button {
    background: #eaf6ff !important;
    border-color: #c7e4f7 !important;
    color: #07539a !important;
}

/* SECTIONS */

.section {
    border: 1px solid #d9eaf5;
    border-radius: 17px;
    padding: 12px 13px;
    margin-bottom: 9px;
    background: #ffffff;
}

.section.pink {
    background: #fffafd;
    border-color: #f2dce7;
}

.section.teal {
    background: #fbfffe;
    border-color: #d8eee9;
}

.section.purple {
    background: #fdfcff;
    border-color: #e5def8;
}

.section-head {
    display: flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 12px;
}

.section-icon {
    width: 36px;
    height: 36px;
    border-radius: 12px;
    background: #edf8ff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    color: #0879d1;
}

.pink .section-icon {
    background: #fff0f6;
    color: #e94884;
}

.teal .section-icon {
    background: #eafaf7;
    color: #0da58f;
}

.purple .section-icon {
    background: #f1edff;
    color: #7054df;
}

.section-title {
    font-size: 14px;
    font-weight: 900;
    color: #07549a;
}

.pink .section-title {
    color: #d84079;
}

.teal .section-title {
    color: #078e7e;
}

.purple .section-title {
    color: #6449ca;
}

.section-desc {
    font-size: 9px;
    color: #7898b1;
    margin-top: 1px;
}

/* FEATURE CARDS */

.feature-card {
    background: #ffffff;
    border: 1px solid #e2eef6;
    border-radius: 13px;
    padding: 9px 11px 7px;
    margin-bottom: 8px;
    box-shadow: 0 4px 12px rgba(35,116,166,.035);
}

.feature-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.feature-name {
    font-size: 9.5px;
    font-weight: 850;
    color: #185d91;
}

.feature-value {
    font-size: 10px;
    font-weight: 900;
    color: #0879d1;
    background: #edf7ff;
    border-radius: 7px;
    padding: 3px 6px;
    margin-top: 5px;
}

.feature-unit {
    font-size: 7.5px;
    color: #7898b1;
}

/* SLIDERS */

div[data-testid="stSlider"] {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

div[data-testid="stSlider"] label {
    display: none !important;
}

div[data-testid="stSlider"] [data-baseweb="slider"] {
    padding-top: 5px !important;
}

div[data-testid="stSelectbox"] label {
    font-size: 9px !important;
    font-weight: 800 !important;
    color: #185d91 !important;
}

/* ANALYSIS */

.analysis-card {
    padding: 25px;
    margin-bottom: 12px;
}

.analysis-title {
    font-size: 20px;
    font-weight: 900;
    color: #07549a;
}

.gauge {
    width: 190px;
    height: 190px;
    margin: 15px auto;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.gauge-inner {
    width: 143px;
    height: 143px;
    border-radius: 50%;
    background: #ffffff;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: 0 5px 20px rgba(20,110,170,.08);
}

.gauge-number {
    font-size: 31px;
    font-weight: 950;
    color: #084d8f;
}

.gauge-label {
    font-size: 10px;
    color: #7697b1;
    margin-top: 5px;
}

.risk {
    padding: 12px;
    border-radius: 14px;
    border: 1px solid #ffd5d5;
    background: #fff5f5;
    display: flex;
    gap: 10px;
}

.risk-badge {
    width: 31px;
    height: 31px;
    min-width: 31px;
    border-radius: 50%;
    background: #ef4444;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
}

.risk-title {
    font-size: 13px;
    font-weight: 900;
    color: #e43d3d;
}

.risk-text {
    font-size: 10px;
    line-height: 1.5;
    color: #7b7f86;
    margin-top: 3px;
}

.disclaimer {
    margin-top: 10px;
    padding: 10px 11px;
    border-radius: 12px;
    background: #edf7ff;
    border: 1px solid #dceefa;
    color: #4c789b;
    font-size: 9.5px;
    line-height: 1.45;
}

.care-message {
    margin-top: 15px;
    padding: 17px;
    border-radius: 16px;
    background: #f8fcff;
    border: 1px solid #dcecf8;
    text-align: center;
}

.care-message h3 {
    color: #07549a;
    margin-bottom: 6px;
}

.care-message p {
    color: #6e8da5;
    font-size: 11px;
    line-height: 1.6;
}

/* HISTORY / ABOUT */

.history-row {
    background: #ffffff;
    border: 1px solid #dcecf8;
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.history-date {
    font-size: 10px;
    color: #7898b1;
}

.history-value {
    font-size: 16px;
    font-weight: 900;
    color: #0879d1;
}

.about-box {
    padding: 25px;
    line-height: 1.7;
    color: #6688a5;
    font-size: 12px;
}

.info-box {
    padding: 11px;
    border: 1px solid #dfe5fa;
    background: #f7f5ff;
    border-radius: 11px;
    font-size: 9px;
    color: #6e6d87;
    line-height: 1.5;
}

/* BUTTON */

.analyze-button button {
    height: 48px !important;
    border-radius: 13px !important;
    background: linear-gradient(
        90deg,
        #0874c9,
        #1599e9
    ) !important;
    color: #ffffff !important;
    border: 0 !important;
    font-weight: 900 !important;
    font-size: 13px !important;
    box-shadow: 0 8px 20px rgba(8,116,201,.18) !important;
}

/* RESPONSIVE */

@media(max-width:1100px) {
    .steps {
        grid-template-columns: 1fr 1fr;
    }
}

@media(max-width:850px) {
    .header-right {
        display: none;
    }
}

#MainMenu,
footer {
    visibility: hidden;
}

</style>
""")

# =========================================================
# NAVIGATION FUNCTIONS
# =========================================================

def go_page(page):
    st.session_state.page = page

    if page == "Clinical Data":
        st.session_state.section = "Patient Data"

    st.rerun()


def go_section(section):
    st.session_state.page = "Clinical Data"
    st.session_state.section = section
    st.rerun()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.html('<div class="nav-logo">🫀</div>')

nav_items = [
    ("⌂", "Home"),
    ("♧", "Clinical Data"),
    ("▥", "Analysis"),
    ("◷", "History"),
    ("ⓘ", "About")
]

for icon, name in nav_items:
    active = "nav-active" if st.session_state.page == name else ""

    st.sidebar.markdown(
        f'<div class="nav-button {active}">',
        unsafe_allow_html=True
    )

    if st.sidebar.button(
        f"{icon}\n{name}",
        key=f"nav_{name}",
        use_container_width=True
    ):
        go_page(name)

    st.sidebar.markdown(
        "</div>",
        unsafe_allow_html=True
    )

st.sidebar.html(
    '<div class="nav-footer">'
    'Better Data<br>'
    'Smarter Predictions<br>'
    'Healthier Tomorrows'
    '</div>'
)

# =========================================================
# HEADER
# =========================================================

st.html("""
<div class="header">
    <div>
        <div class="header-title">
            Sepsis Prediction
        </div>

        <div class="header-sub">
            AI-powered clinical risk assessment
        </div>
    </div>

    <div class="header-right">
        <span>AI</span>
        <span>|</span>

        <div class="avatar">
            ML
        </div>

        <b>Clinical AI</b>
    </div>
</div>
""")

# =========================================================
# HOME
# =========================================================

if st.session_state.page == "Home":

    st.html("""
    <div class="home-card">
        <div class="home-title">
            Welcome to Sepsis Prediction
        </div>

        <div class="home-text">
            An educational machine learning dashboard
            that uses selected clinical measurements
            to estimate the probability of sepsis based
            on a trained Random Forest model.
        </div>

        <br>

        <div style="
            color:#0879d1;
            font-size:12px;
            font-weight:850;
        ">
            AI Clinical Risk Assessment
        </div>
    </div>

    <br>
    """)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.html("""
        <div class="home-feature">
            <div class="home-feature-title">
                🩺 Clinical Data
            </div>

            <div class="home-feature-text">
                Enter the selected patient
                measurements used by the model.
            </div>
        </div>
        """)

    with c2:
        st.html("""
        <div class="home-feature">
            <div class="home-feature-title">
                🤖 ML Prediction
            </div>

            <div class="home-feature-text">
                The Random Forest model analyzes
                the clinical input and produces
                a probability estimate.
            </div>
        </div>
        """)

    with c3:
        st.html("""
        <div class="home-feature">
            <div class="home-feature-title">
                📊 Analysis
            </div>

            <div class="home-feature-text">
                Review the model probability
                and prediction visualization.
            </div>
        </div>
        """)

    st.write("")

    if st.button(
        "Start Clinical Assessment →",
        use_container_width=True
    ):
        go_page("Clinical Data")

# =========================================================
# ANALYSIS PAGE
# =========================================================

elif st.session_state.page == "Analysis":

    prediction = st.session_state.prediction

    if prediction is None:

        st.html("""
        <div class="analysis-card">
            <div class="analysis-title">
                Prediction Analysis
            </div>

            <p style="
                color:#7595ae;
                font-size:12px;
            ">
                No prediction has been generated yet.
                Go to Clinical Data and analyze the
                available measurements first.
            </p>
        </div>
        """)

        if st.button(
            "Go to Clinical Data →",
            use_container_width=True
        ):
            go_page("Clinical Data")

    else:

        probability = prediction * 100
        degree = probability * 3.6

        risk_title = (
            "Possible Sepsis Risk"
            if prediction >= 0.50
            else "Lower Predicted Risk"
        )

        left_col, right_col = st.columns(
            [1, 1.2],
            gap="large"
        )

        # IMAGE

        with left_col:

            if os.path.exists(DOCTOR_PATIENT_IMAGE):
                st.image(
                    DOCTOR_PATIENT_IMAGE,
                    use_container_width=True
                )
            else:
                st.html("""
                <div class="analysis-card"
                     style="min-height:300px;
                            display:flex;
                            align-items:center;
                            justify-content:center;">
                    <div style="
                        text-align:center;
                        color:#7898b1;
                        font-size:13px;
                    ">
                        🩺<br><br>
                        Doctor & Patient
                    </div>
                </div>
                """)

        # RESULT

        with right_col:

            st.html(f"""
            <div class="analysis-card">

                <div class="analysis-title">
                    Prediction Result
                </div>

                <div style="
                    color:#7898b1;
                    font-size:10px;
                ">
                    Latest Random Forest model prediction
                </div>

                <div
                    class="gauge"
                    style="
                        background:
                        conic-gradient(
                            #0879d1 0deg,
                            {degree}deg,
                            #d9edf9 {degree}deg,
                            360deg
                        );
                    "
                >
                    <div class="gauge-inner">

                        <div class="gauge-number">
                            {probability:.2f}%
                        </div>

                        <div class="gauge-label">
                            Model Probability
                        </div>

                    </div>
                </div>

                <div class="risk">

                    <div class="risk-badge">
                        !
                    </div>

                    <div>

                        <div class="risk-title">
                            {risk_title}
                        </div>

                        <div class="risk-text">
                            The model produced a
                            {probability:.2f}% estimated
                            probability based on the
                            submitted clinical features.
                        </div>

                    </div>

                </div>

                <div class="disclaimer">
                    ⓘ This is an educational ML prediction
                    and not a medical diagnosis.
                </div>

            </div>
            """)

            st.progress(prediction)

            st.html("""
            <div class="care-message">

                <h3>
                    Stay Calm, Take Care ♡
                </h3>

                <p>
                    This prediction is only an estimate
                    from a machine learning model.
                    It is not a medical diagnosis.
                    If you have health concerns,
                    please speak with a healthcare
                    professional.
                </p>

            </div>
            """)

# =========================================================
# HISTORY
# =========================================================

elif st.session_state.page == "History":

    st.html("""
    <div class="analysis-card">

        <div class="analysis-title">
            Prediction History
        </div>

        <div style="
            color:#7898b1;
            font-size:10px;
        ">
            Predictions generated during this session.
        </div>

    </div>
    """)

    if not st.session_state.history:

        st.info(
            "No predictions have been generated yet."
        )

    else:

        for item in reversed(
            st.session_state.history
        ):

            st.html(f"""
            <div class="history-row">

                <div>

                    <div style="
                        font-size:12px;
                        font-weight:850;
                        color:#07549a;
                    ">
                        Sepsis Prediction
                    </div>

                    <div class="history-date">
                        {item["time"]}
                    </div>

                </div>

                <div class="history-value">
                    {item["probability"]:.2f}%
                </div>

            </div>
            """)

# =========================================================
# ABOUT
# =========================================================

elif st.session_state.page == "About":

    st.html("""
    <div class="about-box">

        <div style="
            font-size:24px;
            font-weight:900;
            color:#07549a;
        ">
            About the Project
        </div>

        <br>

        This project is an educational machine
        learning application for sepsis risk prediction.

        <br><br>

        <b style="color:#0879d1;">
            Model:
        </b>

        Random Forest Classifier

        <br>

        <b style="color:#0879d1;">
            Selected Input Features:
        </b>

        14 clinical features based on feature importance

        <br>

        <b style="color:#0879d1;">
            Preprocessing:
        </b>

        Median imputation using the trained
        preprocessing artifact

        <br><br>

        The application is designed for learning
        and demonstration purposes. The prediction
        is not a medical diagnosis and should not
        replace professional medical evaluation.

    </div>
    """)

# =========================================================
# CLINICAL DATA
# =========================================================

elif st.session_state.page == "Clinical Data":

    st.html("""
    <div class="panel">

        <div class="clinical-top">

            <div class="clinical-icon">
                ♧
            </div>

            <div>

                <div class="clinical-title">
                    Clinical Data
                </div>

                <div class="clinical-sub">
                    Navigate through the clinical sections
                    and enter the selected measurements.
                </div>

            </div>

        </div>

    </div>
    """)

    # SECTION NAVIGATION

    step_items = [
        ("1", "Patient Data"),
        ("2", "Vital Signs"),
        ("3", "Laboratory Results"),
        ("4", "ICU Context")
    ]

    cols = st.columns(4)

    for i, (number, name) in enumerate(step_items):

        with cols[i]:

            active = (
                "step-active"
                if st.session_state.section == name
                else ""
            )

            st.markdown(
                f'<div class="step-button {active}">',
                unsafe_allow_html=True
            )

            if st.button(
                f"{number}\n{name}",
                key=f"step_{name}",
                use_container_width=True
            ):
                go_section(name)

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

    # =====================================================
    # PATIENT DATA
    # =====================================================

    if st.session_state.section == "Patient Data":

        st.html("""
        <div class="section">

            <div class="section-head">

                <div class="section-icon">
                    ♙
                </div>

                <div>

                    <div class="section-title">
                        Patient Profile
                    </div>

                    <div class="section-desc">
                        Basic patient information used
                        by the model.
                    </div>

                </div>

            </div>
        """)

        c1, c2 = st.columns(2)

        with c1:

            st.html("""
            <div class="feature-card">

                <div class="feature-top">

                    <span class="feature-name">
                        Age
                    </span>

                </div>

                <div class="feature-unit">
                    years
                </div>
            """)

            age = st.slider(
                "Age",
                0,
                120,
                value=st.session_state.age,
                step=1,
                key="age"
            )

            st.html(
                f"""
                <div class="feature-value">
                    {age} years
                </div>
                </div>
                """
            )

        with c2:

            st.html("""
            <div class="feature-card">

                <div class="feature-top">

                    <span class="feature-name">
                        Measurement Hour
                    </span>

                </div>

                <div class="feature-unit">
                    hour of measurement
                </div>
            """)

            hour = st.slider(
                "Hour",
                0,
                24,
                value=st.session_state.hour,
                step=1,
                key="hour"
            )

            st.html(
                f"""
                <div class="feature-value">
                    {hour}:00
                </div>
                </div>
                """
            )

        st.html("</div>")

    # =====================================================
    # VITAL SIGNS
    # =====================================================

    elif st.session_state.section == "Vital Signs":

        st.html("""
        <div class="section pink">

            <div class="section-head">

                <div class="section-icon">
                    ♡
                </div>

                <div>

                    <div class="section-title">
                        Vital Measurements
                    </div>

                    <div class="section-desc">
                        Selected physiological measurements
                        used by the model.
                    </div>

                </div>

            </div>
        """)

        # HEART RATE / TEMPERATURE / PULSE OXIMETRY

        c1, c2, c3 = st.columns(3)

        with c1:

            st.html("""
            <div class="feature-card">

                <div class="feature-top">

                    <span class="feature-name">
                        Heart Rate
                    </span>

                </div>

                <div class="feature-unit">
                    beats/min
                </div>
            """)

            heart_rate = st.slider(
                "Heart Rate",
                20,
                250,
                value=st.session_state.heart_rate,
                step=1,
                key="heart_rate"
            )

            st.html(
                f"""
                <div class="feature-value">
                    {heart_rate} beats/min
                </div>
                </div>
                """
            )

        with c2:

            st.html("""
            <div class="feature-card">

                <div class="feature-top">

                    <span class="feature-name">
                        Temperature
                    </span>

                </div>

                <div class="feature-unit">
                    °C
                </div>
            """)

            temperature = st.slider(
                "Temperature",
                30.0,
                45.0,
                value=st.session_state.temperature,
                step=0.1,
                key="temperature"
            )

            st.html(
                f"""
                <div class="feature-value">
                    {temperature:.1f} °C
                </div>
                </div>
                """
            )

        with c3:

            st.html("""
            <div class="feature-card">

                <div class="feature-top">

                    <span class="feature-name">
                        Pulse Oximetry
                    </span>

                </div>

                <div class="feature-unit">
                    %
                </div>
            """)

            pulse_ox = st.slider(
                "Pulse Oximetry",
                50,
                100,
                value=st.session_state.pulse_ox,
                step=1,
                key="pulse_ox"
            )

            st.html(
                f"""
                <div class="feature-value">
                    {pulse_ox}%
                </div>
                </div>
                """
            )

        # BLOOD PRESSURE

        c1, c2, c3 = st.columns(3)

        with c1:

            st.html("""
            <div class="feature-card">

                <div class="feature-top">

                    <span class="feature-name">
                        Systolic BP
                    </span>

                </div>

                <div class="feature-unit">
                    mmHg
                </div>
            """)

            systolic = st.slider(
                "Systolic BP",
                50,
                250,
                value=st.session_state.systolic,
                step=1,
                key="systolic"
            )

            st.html(
                f"""
                <div class="feature-value">
                    {systolic} mmHg
                </div>
                </div>
                """
            )

        with c2:

            st.html("""
            <div class="feature-card">

                <div class="feature-top">

                    <span class="feature-name">
                        Diastolic BP
                    </span>

                </div>

                <div class="feature-unit">
                    mmHg
                </div>
            """)

            diastolic = st.slider(
                "Diastolic BP",
                20,
                150,
                value=st.session_state.diastolic,
                step=1,
                key="diastolic"
            )

            st.html(
                f"""
                <div class="feature-value">
                    {diastolic} mmHg
                </div>
                </div>
                """
            )

        with c3:

            st.html("""
            <div class="feature-card">

                <div class="feature-top">

                    <span class="feature-name">
                        Mean Arterial Pressure
                    </span>

                </div>

                <div class="feature-unit">
                    mmHg
                </div>
            """)

            map_value = st.slider(
                "Mean Arterial Pressure",
                30,
                200,
                value=st.session_state.map_value,
                step=1,
                key="map_value"
            )

            st.html(
                f"""
                <div class="feature-value">
                    {map_value} mmHg
                </div>
                </div>
                """
            )

        # RESPIRATION / FIO2

        c1, c2 = st.columns(2)

        with c1:

            st.html("""
            <div class="feature-card">

                <div class="feature-top">

                    <span class="feature-name">
                        Respiration Rate
                    </span>

                </div>

                <div class="feature-unit">
                    breaths/min
                </div>
            """)

            respiration = st.slider(
                "Respiration Rate",
                5,
                80,
                value=st.session_state.respiration,
                step=1,
                key="respiration"
            )

            st.html(
                f"""
                <div class="feature-value">
                    {respiration} breaths/min
                </div>
                </div>
                """
            )

        with c2:

            st.html("""
            <div class="feature-card">

                <div class="feature-top">

                    <span class="feature-name">
                        FiO₂
                    </span>

                </div>

                <div class="feature-unit">
                    %
                </div>
            """)

            fio2 = st.slider(
                "FiO₂",
                21,
                100,
                value=st.session_state.fio2,
                step=1,
                key="fio2"
            )

            st.html(
                f"""
                <div class="feature-value">
                    {fio2}%
                </div>
                </div>
                """
            )

        st.html("</div>")

    # =====================================================
    # LABORATORY
    # =====================================================

    elif st.session_state.section == "Laboratory Results":

        st.html("""
        <div class="section teal">

            <div class="section-head">

                <div class="section-icon">
                    ⚗
                </div>

                <div>

                    <div class="section-title">
                        Laboratory Measurements
                    </div>

                    <div class="section-desc">
                        Selected laboratory values used
                        by the model.
                    </div>

                </div>

            </div>
        """)

        c1, c2, c3 = st.columns(3)

        with c1:

            st.html("""
            <div class="feature-card">

                <div class="feature-top">

                    <span class="feature-name">
                        pH
                    </span>

                </div>

                <div class="feature-unit">
                    blood pH
                </div>
            """)

            ph = st.slider(
                "pH",
                6.5,
                8.0,
                value=st.session_state.ph,
                step=0.01,
                key="ph"
            )

            st.html(
                f"""
                <div class="feature-value">
                    {ph:.2f}
                </div>
                </div>
                """
            )

        with c2:

            st.html("""
            <div class="feature-card">

                <div class="feature-top">

                    <span class="feature-name">
                        PaCO₂
                    </span>

                </div>

                <div class="feature-unit">
                    mmHg
                </div>
            """)

            paco2 = st.slider(
                "PaCO₂",
                10,
                150,
                value=st.session_state.paco2,
                step=1,
                key="paco2"
            )

            st.html(
                f"""
                <div class="feature-value">
                    {paco2} mmHg
                </div>
                </div>
                """
            )

        with c3:

            st.html("""
            <div class="feature-card">

                <div class="feature-top">

                    <span class="feature-name">
                        Excess Bicarbonate
                    </span>

                </div>

                <div class="feature-unit">
                    mmol/L
                </div>
            """)

            bicarbonate = st.slider(
                "Excess Bicarbonate",
                -30.0,
                30.0,
                value=st.session_state.bicarbonate,
                step=0.1,
                key="bicarbonate"
            )

            st.html(
                f"""
                <div class="feature-value">
                    {bicarbonate:.1f} mmol/L
                </div>
                </div>
                """
            )

        st.html("</div>")

    # =====================================================
    # ICU CONTEXT
    # =====================================================

    elif st.session_state.section == "ICU Context":

        st.html("""
        <div class="section purple">

            <div class="section-head">

                <div class="section-icon">
                    ▣
                </div>

                <div>

                    <div class="section-title">
                        ICU Context
                    </div>

                    <div class="section-desc">
                        MICU administrative identifier
                        used by the model.
                    </div>

                </div>

            </div>
        """)

        c1, c2 = st.columns([1, 1])

        with c1:

            icu_unit = st.selectbox(
                "MICU Status",
                ["None", "MICU"],
                index=[
                    "None",
                    "MICU"
                ].index(
                    st.session_state.icu_unit
                    if st.session_state.icu_unit in
                    ["None", "MICU"]
                    else "None"
                ),
                key="icu_unit"
            )

        with c2:

            st.html("""
            <div class="info-box">

                <b style="color:#6249c9;">
                    ⓘ MICU Identifier
                </b>

                <br>

                Select MICU if the patient is
                associated with the Medical ICU.

            </div>
            """)

        st.html("</div>")

        # READY TO ANALYZE

        st.html("""
        <div style="
            padding:12px;
            border-radius:14px;
            background:#eef8ff;
            border:1px solid #d9ecf8;
            color:#5d82a0;
            font-size:10px;
            line-height:1.5;
        ">

            <b style="color:#0879d1;">
                Ready to analyze?
            </b>

            <br>

            Review the selected clinical values,
            then click the button below to generate
            the model prediction.

        </div>
        """)

        st.write("")

        # =================================================
        # PREDICTION
        # =================================================

        if st.button(
            "Analyze Clinical Data →",
            use_container_width=True
        ):

            try:

                # Create all model features as NaN

                values_dict = {
                    feature: np.nan
                    for feature in feature_names
                }

                # User inputs

                values_dict["Hour"] = (
                    st.session_state.hour
                )

                values_dict[
                    "Heart_rate__in_beats_per_minute"
                ] = (
                    st.session_state.heart_rate
                )

                values_dict[
                    "Respiration_rate__breaths_per_minute"
                ] = (
                    st.session_state.respiration
                )

                values_dict[
                    "Temperature__deg_C"
                ] = (
                    st.session_state.temperature
                )

                values_dict[
                    "Mean_arterial_pressure__mm_Hg"
                ] = (
                    st.session_state.map_value
                )

                values_dict[
                    "Fraction_of_inspired_oxygen__percentage"
                ] = (
                    st.session_state.fio2
                )

                values_dict[
                    "Systolic_BP__mm_Hg"
                ] = (
                    st.session_state.systolic
                )

                values_dict[
                    "Age__years"
                ] = (
                    st.session_state.age
                )

                values_dict[
                    "Diastolic_BP__mm_Hg"
                ] = (
                    st.session_state.diastolic
                )

                values_dict["pH"] = (
                    st.session_state.ph
                )

                values_dict[
                    "Partial_pressure_of_carbon_dioxide_from_arterial_blood__mm_Hg"
                ] = (
                    st.session_state.paco2
                )

                values_dict[
                    "Excess_bicarbonate__mmol_L"
                ] = (
                    st.session_state.bicarbonate
                )

                values_dict[
                    "Pulse_oximetry__percentage"
                ] = (
                    st.session_state.pulse_ox
                )

                values_dict[
                    "Administrative_identifier_for_ICU_unit__MICU___false__0__or_true"
                ] = (
                    1
                    if st.session_state.icu_unit == "MICU"
                    else 0
                )

                # Create DataFrame
                # in exact model feature order

                input_df = pd.DataFrame(
                    [values_dict],
                    columns=feature_names
                )

                # Apply trained imputer

                input_imputed = imputer.transform(
                    input_df
                )

                # Prediction

                probability = float(
                    model.predict_proba(
                        input_imputed
                    )[0][1]
                )

                # Save prediction

                st.session_state.prediction = probability

                # Save history

                st.session_state.history.append(
                    {
                        "time": datetime.now().strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                        "probability": probability * 100
                    }
                )

                # Go to Analysis

                st.session_state.page = "Analysis"

                st.rerun()

            except Exception as e:

                st.error(
                    "Something went wrong while generating "
                    "the prediction."
                )

                st.exception(e)