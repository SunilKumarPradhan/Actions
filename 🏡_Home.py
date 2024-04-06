import os

import streamlit as st
import streamlit_analytics

from utils import add_logo_to_sidebar, add_footer, add_email_signup_form

streamlit_analytics.start_tracking()

st.set_page_config(
    page_title="Legal AI Demos",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'sunilkumarweb47@gmail.com',
        'Report a bug': None,
        'About': "## This showcases LegalConnect India in Actions"
    }
)

add_logo_to_sidebar()

st.title("👋 Welcome - LegalConnect from Team Hacktivist ")

# st.sidebar.success("👆 Select a demo above.")

st.markdown(
    """
    Here you will find demos for the most common Legal AI Actions including:
    - 🏷 **Label** - Use AI to **label** text
    - ⛏ **Extract** - Use AI to **extract** information from text
    - 🔗 **Compare** - Use AI to **compare** passages of text
    - 🗂 **Organise** - Use AI to **organise** a collection of texts
    - 🔎 **Find** - Use AI to **find** relevant information from a collection of texts
    - 💬 **Chat** - Use AI to **chat**
    - 📝 **Summarise** - Use AI to **summarise** text
    """)

st.warning(" The AI models have not been optimised for prediction performance.")

st.info("#### 👈 Select a feature from the sidebar to use them")

add_email_signup_form()

st.markdown(
    """
        🏛️ Exploring Legal Solutions with Legal Connect India

Exploring legal solutions is akin to crafting a bespoke legal strategy. At Legal Connect India, our legal experts 🧑‍⚖️ combine the essential legal elements 📜 (laws, regulations, and case precedents) to construct innovative legal solutions tailored to your needs.

Our legal professionals act as expert 🧑‍🍳 chefs, crafting these legal strategies with the finest legal ingredients 📜. Just like a chef combines ingredients to create a delicious meal, we integrate legal knowledge and expertise to address your legal challenges effectively.

Once our legal strategy is ready, it's served to you, the client, 🛎️ ready for implementation. Our mission at Legal Connect India is to empower legal professionals like you to navigate legal complexities with confidence and clarity.

🌐 About Legal Connect India

Legal Connect India specializes in providing comprehensive legal solutions and services. We're dedicated to enabling legal professionals to embrace technology and collaboration to enhance legal outcomes.

Our team comprises skilled legal experts and advisors who are committed to delivering high-quality legal support. Let Legal Connect India be your trusted partner in navigating the legal landscape effectively and efficiently.
    """
)

add_footer()

streamlit_analytics.stop_tracking(unsafe_password=os.environ["ANALYTICS_PASSWORD"])