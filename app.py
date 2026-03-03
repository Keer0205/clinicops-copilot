import os
import streamlit as st
from openai import OpenAI

# --- Load key from Streamlit Secrets first, then env ---
api_key = st.secrets.get("OPENAI_API_KEY", None)
if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")

# --- Clean the key (removes hidden newline/spaces) ---
if api_key:
    api_key = api_key.strip().replace("\n", "").replace("\r", "")

# --- Safe debug (does NOT show the key) ---
st.sidebar.caption(f"✅ Key loaded: {bool(api_key)} | Length: {len(api_key) if api_key else 0}")

if not api_key:
    st.error("OPENAI_API_KEY missing. Go to Manage app → Settings → Secrets and set it.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- Quick auth test (will give clearer error) ---
try:
    client.models.list()
except Exception as e:
    st.error("❌ OpenAI auth failed. Your API key is missing/invalid or not applied yet.")
    st.code(str(e))
    st.stop()
