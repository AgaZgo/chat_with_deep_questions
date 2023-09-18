import streamlit as st


def clear_state():
    for k in st.session_state:
        del st.session_state[k]
        




