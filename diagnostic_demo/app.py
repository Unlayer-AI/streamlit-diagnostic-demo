import streamlit as st

# from streamlit_card import card as st_card
import pandas as pd
import pickle
import time


from app_utils.page_view import format_style
from actions import (
    run_calibration,
    display_calibration,
    run_fairness,
    display_fairness,
    run_attribution,
    display_attribution,
    run_simpler_model,
    display_simpler_model,
)

st.set_page_config(
    page_title="Unlayer AI - diagnostic demo",
    page_icon="favicon.ico",
    # layout="wide",
)
format_style()

css = """
.uploadedFiles {
    display: none;
}
"""

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
st.markdown(
    """
<h1 class="logo pulse">
    <span class="gradient-text">Unlayer</span> <span class="text-white">AI</span>
    <span class="text-white" style="font-size: 2rem;">- diagnostic demo</span>
</h1> 
""",
    unsafe_allow_html=True,
)

init_description = st.caption(
    """This demo runs explainable AI checks on a binary classification model and dataset,
expecting a label with values `1` for positive class and `0` for negative class.

This public demo operates on a pre-loaded
random forest classifier and prepared dataset derived from the ["Adult" (Census Income) dataset](https://doi.org/10.24432/C5XW20)
by Barry Becker & Ronny Kohavi from UCI Machine Learning Repository, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode).
See [this colab](https://colab.research.google.com/drive/1x7xN5iMd3BHJOyxe64QfRYUN_-VtVuJs?usp=sharing) for more details.
""",
    unsafe_allow_html=True,
)


btn_accept_disclaimer = None
if (
    not hasattr(st.session_state, "disclaimer_accepted")
    or not st.session_state.disclaimer_accepted
):
    disclaimer_container = st.container(border=True)
    with disclaimer_container:
        st.markdown(
            """\
### DISCLAIMER
This demo is provided <strong>for educational and informational purposes only</strong>. It is not intended as legal, ethical, or professional advice and should <strong>not be relied upon</strong> for making decisions of any kind, including those related to fairness, compliance, or deployment of machine learning models.
All outputs, analyses, and recommendations are provided <strong>"as is"</strong> with <strong>no warranties</strong>, express or implied, regarding accuracy, completeness, performance, or fitness for any particular purpose.
By using this tool, you acknowledge and agree that: You are solely responsible for evaluating the results and for any actions you take based on them; The developers and contributors are <strong>not liable</strong> for any damages or losses, direct or indirect, arising from the use of this software.
<br>
For transparency, this project is open source. You may review the source code at:
<a href="https://github.com/unlayer-ai/streamlit-diagnostic-demo/" target="_blank">GitHub Repository</a>
""",
            unsafe_allow_html=True,
        )
        _, col = st.columns([4, 3])
        with col:
            btn_accept_disclaimer = st.button(
                "I read the disclaimer and wish to proceed"
            )
    

if btn_accept_disclaimer:
    st.session_state.disclaimer_accepted = True
    disclaimer_container.empty()
    st.rerun()

footer = """<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: black;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    z-index: 9999;
}
</style>
<div class="footer">
<!-- gray colored text, centered, some margin top -->
<p style="color: gray; text-align: center; margin-top: 1rem; font-size: 0.8rem;">
© 2025 Unlayer AI. All rights reserved.
</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)


def run_app():
    
    df_header_placeholder = st.empty()
    df_placeholder = st.empty()
    var_to_predict_placeholder = st.empty()
    llm_option_expander = st.empty()

    col1, col2 = st.columns([4, 2])

    with col2:
        btn_diagnose = None
        btn_diagnose_placeholder = st.empty()

    # Always use the bundled demo model and dataset
    if not hasattr(st.session_state, "model") or not hasattr(st.session_state, "df"):
        with open("demo_data/model.pkl", "rb") as model_file:
            st.session_state.model = pickle.load(model_file)
        st.session_state.df = pd.read_csv("demo_data/dev.csv")

    # hide intro once demo data is ready
    init_description.empty()

    # show preview and allow selecting the target variable
    df_header_placeholder.caption(
        f"""
            Loaded a *<span class=\"text-white\">{st.session_state.model.__class__.__name__}</span>*
            to be evaluated on *Adult (Census Income)* dev set - preview:
    """,
        unsafe_allow_html=True,
    )
    df_placeholder.write(st.session_state.df.head(5))

    def set_desirable_class():
        values = sorted(
            st.session_state.df[st.session_state.target_variable].unique()
        )
        st.session_state.target_class = int(values[-1])
        print(
            f"desirable class for {st.session_state.target_variable} is {st.session_state.target_class}"
        )

    # get the variable to predict
    st.session_state.target_variable = str(
        st.session_state.df.columns[len(st.session_state.df.columns) - 1]
    )
    set_desirable_class()
    st.session_state.target_variable = var_to_predict_placeholder.selectbox(
        "Select the variable to predict (binary class expected, with 1=desirable, 0=undesirable):",
        st.session_state.df.columns,
        index=len(st.session_state.df.columns) - 1,
        on_change=set_desirable_class,
    )

    with col2:
        btn_diagnose = btn_diagnose_placeholder.button("🩺 Diagnose")

    # if model is uploaded and btn "diagnose" is clicked
    def run_diagnostic(
        df_header_placeholder,
        df_placeholder,
        var_to_predict_placeholder,
        btn_diagnose_placeholder,
    ):
        st.toast("Diagnosis started", icon="🩺")
        # if not st.session_state.df or not st.session_state.model:
        #    st.error("Please upload a dataset and a model")
        #    st.stop()
        df_header_placeholder.empty()
        df_placeholder.empty()
        var_to_predict_placeholder.empty()
        llm_option_expander.empty()
        btn_diagnose_placeholder.empty()

        ATTRIBUTION_SUBSAMPLE = 20

        # placeholder
        with st.spinner("Considering calibration..."):
            time.sleep(2)
            calibration_result = run_calibration()
        with st.spinner("Running fairness checks (~a few seconds)..."):
            fairness_result = run_fairness()
        with st.spinner("Estimating feature attribution (~a dozen seconds)..."):
            attribution_result = run_attribution(subsample_no=ATTRIBUTION_SUBSAMPLE)
        with st.spinner("Attempting simpler modeling (~a few minutes)..."):
            simpler_model_result = run_simpler_model()
            
        calib_col, fair_col, attr_col, surr_col = st.columns([1, 1, 1, 1])
        calib_tab, fair_tab, attr_tab, surr_tab, contact_tab = st.tabs(
            ["Calibration", "Fairness", "Attribution", "Simpler modeling", "Contact us"]
        )

        display_calibration(calib_col, calib_tab, calibration_result)
        display_fairness(fair_col, fair_tab, fairness_result)
        display_attribution(
            attr_col, attr_tab, attribution_result, subsample_no=ATTRIBUTION_SUBSAMPLE
        )
        display_simpler_model(surr_col, surr_tab, simpler_model_result)

        contact_tab.markdown(
            """\
    Looking to build AI that your team and stakeholders can trust?
    We’re here to help you across a wide range of challenges:

    #### Transparency & Fairness
    - 🔍 Enhance transparency with interpretable models and feature attribution
    - ⚖️ Increase fairness through bias detection and mitigation strategies

    #### Actionability & Privacy
    - 🔄 Empower your users with recourse via counterfactual explanations
    - 🔒 Increase privacy with synthetic data generation
    
    #### Robustness & Causality
    - 🧩 Estimate cause–effect relationships with causal inference and propensity scoring  
    - ⚙️ Improve robustness using adversarial training methods
    
    #### Compliance & Research
    - 📑 Answer regulatory questions with clear documentation and compliance checks
    - ✍️ Support research with expert guidance and collaboration

    [Get in touch](https://unlayer.ai/#contact) to explore how we can help \
    you turn responsible AI principles into systems that are transparent, reliable, and trusted.
    """
)

    if btn_diagnose:
        # check that the class is OK
        class_values = sorted(
            st.session_state.df[st.session_state.target_variable].unique()
        )
        if len(class_values) != 2:
            st.error(
                f"Target {st.session_state.target_variable} is not binary (0-1). \
                This demo works only with binary labels."
            )
        else:
            run_diagnostic(
                df_header_placeholder,
                df_placeholder,
                var_to_predict_placeholder,
                btn_diagnose_placeholder,
            )


if (
    hasattr(st.session_state, "disclaimer_accepted")
    and st.session_state.disclaimer_accepted
):
    run_app()
