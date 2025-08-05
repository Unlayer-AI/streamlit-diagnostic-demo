import streamlit as st

# from streamlit_card import card as st_card
import pandas as pd
import pickle
import time
import os


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
st.markdown("### Unlayer AI - diagnostic demo")

btn_diagnose = None
model_uploader = st.empty()
data_uploader = st.empty()
df_placeholder = st.empty()
var_to_predict_placeholder = st.empty()
target_class_placeholder = st.empty()
llm_option_expander = st.empty()

col1, col2 = st.columns([4, 2])

with col2:
    use_demo_model_btn_placeholder = st.empty()
    use_demo_data_btn_placeholder = st.empty()
    btn_diagnose_placeholder = st.empty()

demo_text_holder = st.empty()

uploaded_model = model_uploader.file_uploader(
    "1️⃣ Upload a scikit-learn compatible binary classifier", type=["pkl"]
)
use_demo_model = use_demo_model_btn_placeholder.button(
    "or use demo classifier", use_container_width=True
)
demo_text_holder.html(
    """<br>
<p style="font-size: 14px; color: rgba(255, 255, 255, 0.6);">
    A Python notebook to get a demo dataset and classifier is available
        <a target="_blank" rel="noopener noreferrer"
        href="https://colab.research.google.com/drive/1x7xN5iMd3BHJOyxe64QfRYUN_-VtVuJs?usp=sharing">here (Ctrl/Cmd+Click)</a>
</p>
"""
)

if uploaded_model or use_demo_model or hasattr(st.session_state, "model"):
    if hasattr(st.session_state, "model"):
        uploaded_model = st.session_state.model

    if not hasattr(st.session_state, "model"):
        if use_demo_model:
            uploaded_model = open("demo_data/model.pkl", "rb")

        uploaded_model = pickle.load(uploaded_model)

    # check the model is OK
    if not hasattr(uploaded_model, "predict"):
        st.error(
            "The uploaded model has no `predict` function.\n"
            + "Please upload a compatible model."
        )
        uploaded_model = None

    elif not hasattr(uploaded_model, "predict_proba"):
        st.error(
            "The uploaded model has no `predict_proba` function.\n"
            + "Please upload a compatible model."
        )
        uploaded_model = None

    else:
        st.toast("Classifier loaded", icon="🤖")
        # hide model uploader control
        st.session_state.model = uploaded_model
        model_uploader.empty()
        use_demo_model_btn_placeholder.empty()

        # Move to next step: data upload
        uploaded_data = data_uploader.file_uploader(
            "2️⃣ Upload a binary classification development set", type=["csv"]
        )
        use_demo_data = use_demo_data_btn_placeholder.button(
            "or use demo dataset", use_container_width=True
        )

        if uploaded_data or use_demo_data or hasattr(st.session_state, "df"):
            if use_demo_data:
                uploaded_data = "demo_data/dev.csv"

            # open dataset and extract columns
            if not hasattr(st.session_state, "df"):
                st.session_state.df = pd.read_csv(uploaded_data)
            else:
                uploaded_data = st.session_state.df

            st.toast("Development set loaded", icon="🗂️")
            data_uploader.empty()
            use_demo_data_btn_placeholder.empty()
            demo_text_holder.empty()

            # show
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
                "What is the variable to predict? (binary class expected, with 1=desirable, 0=undesirable)",
                st.session_state.df.columns,
                index=len(st.session_state.df.columns) - 1,
                on_change=set_desirable_class,
            )

            expander = llm_option_expander.expander(
                "Use LLM for advanced diagnostics",
                expanded=st.session_state.get("llm_option_expander", False),
            )
            with expander:
                st.session_state.llm_option_expander = True
                st.session_state.llm_api_key = st.text_input(
                    "LLM API key (e.g., OpenAI API key)",
                    placeholder="sk-...",
                )
                st.session_state.llm_model = st.text_input(
                    "LLM model name",
                    placeholder="gpt-4o",
                )

            with col2:
                btn_diagnose = btn_diagnose_placeholder.button("🩺 Diagnose")


# if model is uploaded and btn "diagnose" is clicked
def run_diagnostic(
    data_uploader, df_placeholder, var_to_predict_placeholder, btn_diagnose_placeholder
):
    st.toast("Diagnosis started", icon="🩺")
    # if not st.session_state.df or not st.session_state.model:
    #    st.error("Please upload a dataset and a model")
    #    st.stop()
    data_uploader.empty()
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
        * Want to learn more about the checks we perform or how to address the issues they uncover?
        * Curious about other aspects of responsible AI?
            * Counterfactual explanations, integrated gradients, causality, symbolic regression...
            * ...or assessing generative models, tracing multi-agent systems?
        
        We are here to help! Reach out to us at [Unlayer AI](https://unlayer.ai/#contact) to discuss your needs and how
        we can assist you in building responsible AI systems."""
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
            data_uploader,
            df_placeholder,
            var_to_predict_placeholder,
            btn_diagnose_placeholder,
        )


footer = """<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: black;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}
</style>
<div class="footer">
<p style="text-align: center; font-size: 10px; color: rgba(255, 255, 255, 0.6); padding-top: 12px;">
<strong>DISCLAIMER</strong><br>
  This demo is provided <strong>for educational and informational purposes only</strong>. It is not intended to serve as legal, ethical, or professional advice, and should <strong>not be relied upon</strong> for making decisions of any kind, including those related to fairness, compliance, or deployment of machine learning models.
  All outputs, analyses, and recommendations generated by this tool are provided <strong>"as is"</strong> with <strong>no warranties</strong>, express or implied, regarding accuracy, completeness, performance, or fitness for any particular purpose.
  By using this tool, you acknowledge and agree that: 
  You are solely responsible for evaluating the results and for any actions you take based on them;
  You will <strong>not upload any confidential, sensitive, or personally identifiable information</strong>;
  Any models, datasets, API keys, or other information you submit are used <strong>at your own risk</strong>, and may be stored or processed for demonstration purposes only;
  The developers and contributors of this tool are <strong>not liable</strong> for any damages or losses, direct or indirect, arising from the use of this software.
  <br>
  This project is open source. You may review the source code at:
  <a href="https://github.com/unlayer-ai/streamlit-diagnostic-demo/" target="_blank">GitHub Repository</a>
  <br>
© 2025 Unlayer AI. All rights reserved.
</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
