import os
import streamlit as st
from openai import OpenAI

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY missing. Add it in Streamlit Cloud → Manage app → Settings → Secrets.")
    st.stop()

api_key = api_key.strip().replace("\n", "").replace("\r", "")

client = OpenAI(api_key=api_key)
