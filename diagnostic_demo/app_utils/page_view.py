import streamlit as st


def _set_sidebar_right():
    html = """
    <style>
        .reportview-container {
        flex-direction: row-reverse;
        }

        header > .toolbar {
        flex-direction: row-reverse;
        left: 1rem;
        right: auto;
        }

        .sidebar .sidebar-collapse-control,
        .sidebar.--collapsed .sidebar-collapse-control {
        left: auto;
        right: 0.5rem;
        }

        .sidebar .sidebar-content {
        transition: margin-right .3s, box-shadow .3s;
        }

        .sidebar.--collapsed .sidebar-content {
        margin-left: auto;
        margin-right: -21rem;
        }

        @media (max-width: 991.98px) {
        .sidebar .sidebar-content {
            margin-left: auto;
        }
        }
    </style>
    """
    st.markdown(html, unsafe_allow_html=True)


def _hide_deploy_button():
    html = """
        <style>
        .stDeployButton {
            visibility: hidden;
        }
        </style>
        """
    st.markdown(html, unsafe_allow_html=True)

def _set_css():
    with open("diagnostic_demo/style/style.css") as css:
        st.markdown( f'<style>{css.read()}</style>', unsafe_allow_html= True)

def format_style():
    #st.set_page_config(layout="wide")
    _set_css()
    _hide_deploy_button()
    _set_sidebar_right()
