"""
ClinicOps Copilot — Share Demo page.

Provides a ready-to-copy outreach message for clinic owners.
Kept separate from the main Q&A app so it doesn't clutter the clinical workflow.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Share Demo — ClinicOps Copilot", page_icon="📣", layout="centered")
st.title("📣 Share This Demo")
st.caption("Copy the message below and send it to clinic owners via WhatsApp or email.")

# Prefill with the public URL if configured in Streamlit Secrets
default_url = st.secrets.get("APP_URL", "")
demo_url = st.text_input(
    "Your demo URL",
    value=default_url,
    placeholder="https://<your-app>.streamlit.app",
    help="Paste your Streamlit Cloud URL here. You can also set APP_URL in Streamlit Secrets to prefill this automatically.",
)

pitch = f"""Hi Doctor 👋

I built a small assistant for clinics: **ClinicOps Copilot**.

✅ Upload your clinic PDFs (SOPs, consent, aftercare, pricing)
✅ Staff can ask questions and get answers with **page citations**
✅ If it's not in your documents, it **refuses** — no guessing
✅ Basic monitoring: p50/p95 latency + downloadable CSV logs

Demo link: {demo_url if demo_url else "[paste demo link here]"}

If you want, I can set this up for your clinic for a quick trial."""

st.text_area("Copy message (WhatsApp / Email)", value=pitch, height=240)

safe_pitch = (
    pitch
    .replace("&", "&amp;")
    .replace("<", "&lt;")
    .replace(">", "&gt;")
)

components.html(
    f"""
    <button id="copybtn" style="
        padding: 8px 16px;
        border-radius: 8px;
        border: 1px solid #ccc;
        background: #fff;
        cursor: pointer;
        font-size: 14px;
    ">
      📋 Copy message to clipboard
    </button>
    <span id="status" style="margin-left: 10px; color: #2e7d32; font-size: 14px;"></span>
    <textarea id="pitch" style="position: absolute; left: -9999px;">{safe_pitch}</textarea>
    <script>
      document.getElementById("copybtn").onclick = async () => {{
        const txt = document.getElementById("pitch").value;
        const status = document.getElementById("status");
        try {{
          await navigator.clipboard.writeText(txt);
          status.textContent = "Copied ✅";
          setTimeout(() => status.textContent = "", 2500);
        }} catch (e) {{
          status.textContent = "Copy failed — please select and copy manually.";
        }}
      }};
    </script>
    """,
    height=55,
)

st.caption("Tip: Set APP_URL in Streamlit Secrets to prefill the demo link automatically.")
