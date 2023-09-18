import streamlit as st
import os

from utils.helpers import clear_state


def sidebar():
    with st.sidebar:
        st.session_state['episode'] = st.selectbox(
            'Choose the episode you want to chat with:', 
            os.listdir('data/') + ['all'], 
            index=0, 
            on_change=clear_state
        )
        
        st.session_state['gpt_model'] = st.selectbox(
            'Choose GPT model:', 
            ['gpt-3.5-turbo', 'gpt-4'], 
            index=0,
            on_change=clear_state
        )
        
        openai_api_key = st.text_input(
            "Your OpenAI API key",
        )

        if openai_api_key:    
            os.environ['OPENAI_API_KEY'] = openai_api_key
        else:
            st.warning(
                "You need OpenAI API key use this chat. You can get a key at"
                " https://platform.openai.com/account/api-keys."
            )
            st.stop()