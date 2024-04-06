import os
import joblib

import plotly.graph_objects as go
import streamlit_analytics

from huggingface_hub import hf_hub_download

import streamlit as st
import streamlit.components.v1 as components

from lime.lime_text import LimeTextExplainer

from utils import add_logo_to_sidebar, add_footer

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "simplexico/cuad-sklearn-clause-classifier"
FILENAME = "CUAD-clause-classifier.pkl"

EXAMPLE_TEXT = """This Agreement and any dispute or claim arising out of or in connection with it 
or its subject matter or formation (including non-contractual disputes or claims) shall be 
governed by and construed in accordance with the law of England."""

streamlit_analytics.start_tracking()

## Layout stuff
st.set_page_config(
    page_title="Label Clause Demo",
    page_icon="🏷",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:hello@simplexico.ai',
        'Report a bug': None,
        'About': "## This a demo showcasing different Legal AI Actions"
    }
)

add_logo_to_sidebar()

st.title('🏷 Label Clause Demo')
st.write("""
This demo shows how AI can be used to label text.
We've trained an AI model to label a clause by its clause type.
""")

@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load(
        hf_hub_download(repo_id=REPO_ID, filename=FILENAME, token=HF_TOKEN)
    )
    return model


@st.cache(allow_output_mutation=True)
def get_prediction_prob(text):
    y_pred = model.predict([text])[0]
    y_probs = model.predict_proba([text])[0]
    return y_pred, y_probs


st.markdown('### 🖊 Enter clause text')
text = st.text_area(label='**Enter Clause Text**', label_visibility='collapsed', value=EXAMPLE_TEXT, height=100)
button = st.button('**Label Clause**', type='primary', use_container_width=True)

with st.spinner('⚙️ Loading model...'):
    model = load_model()

classes = [s.upper() for s in model.classes_]

if button:
    with st.spinner('⚙️ Processing Clause...'):
        y_pred, y_probs = get_prediction_prob(text)
        explainer = LimeTextExplainer(class_names=[cls[:9] + '…' for cls in model.classes_])
        exp = explainer.explain_instance(text,
                                         model.predict_proba,
                                         num_features=10,
                                         top_labels=1)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('### 🤖 Prediction Results')
        st.write(
            f"The model predicts that this is a **{y_pred}** clause with **{y_probs.max() * 100:.2f}%** confidence.")

        fig = go.Figure(go.Bar(
            x=y_probs * 100,
            y=model.classes_,
            orientation='h'))
        fig.update_layout(
            title="Model Confidence",
            xaxis_title="Confidence (%)",
            yaxis_title="Clause Type",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('### 🔮 Prediction Explainability')
        st.write(
            'We can perform an analysis to work out what terms in the clause were most important in deciding the predicted clause type:')
        components.html(exp.as_html(predict_proba=False), height=600)

    add_footer()

streamlit_analytics.stop_tracking(unsafe_password=os.environ["ANALYTICS_PASSWORD"])