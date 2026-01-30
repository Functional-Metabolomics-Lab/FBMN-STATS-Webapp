import streamlit as st
import pandas as pd

from src.common import *

page_setup('Chat', sidebar_chat=False)  # Disable sidebar to prevent redundancy

st.markdown("# Chat")

gemini_chat(descending=True)  # Show messages in chronological order top to bottom
