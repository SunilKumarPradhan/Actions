import os
from io import StringIO

import joblib

from copy import deepcopy
from pypdf import PdfReader
import pandas as pd
import plotly.express as px

from huggingface_hub import hf_hub_download, snapshot_download

import streamlit as st
import streamlit_analytics
from utils import add_logo_to_sidebar, add_footer, add_email_signup_form

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_REPO_ID = "simplexico/cuad-sklearn-contract-clustering"
DATA_REPO_ID = "simplexico/cuad-top-ten"
MODEL_FILENAME = "cuad_tfidf_umap_kmeans.pkl"
DATA_FILENAME = "cuad_top_ten_popular_contract_types.json"

streamlit_analytics.start_tracking()

st.set_page_config(
    page_title="Organise Demo",
    page_icon="🗂",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:hello@simplexico.ai',
        'Report a bug': None,
        'About': "## This a demo showcasing different Legal AI Actions"
    }
)

add_logo_to_sidebar()

st.title('🗂 Organise Demo')
st.write("""
This demo shows how AI can be used to organise a collection of texts.
We've trained a model to group documents into similar types.
The plot below shows a sample set of contracts that have been automatically grouped together.
Each point in the plot represents how the model interprets a contract, the closer together a pair of points are, the more similar they appear to the model.
Similar documents are grouped by color.
\n**TIP:** Hover over each point to see the filename of the contract. Groups can be added or removed by clicking on the symbol in the plot legend.
""")

st.info("**👈 Upload your own documents on the left (as .txt or .pdf files)**")


@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load(
        hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME, token=HF_TOKEN)
    )
    return model


@st.cache(allow_output_mutation=True)
def load_dataset():
    snapshot_download(repo_id=DATA_REPO_ID, token=HF_TOKEN, local_dir='./', repo_type='dataset')
    df = pd.read_json(DATA_FILENAME)
    return df


def get_transform_and_predictions(model, X):
    y = model.predict(X)
    X_transform = model[:2].transform(X)
    return X_transform, y


def generate_plot(X, y, filenames):
    fig = px.scatter_3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        color=[str(y_i) for y_i in y], hover_name=filenames)

    fig.update_traces(
        marker_size=8,
        marker_line=dict(width=2),
        selector=dict(mode='markers')
    )

    fig.update_layout(
        legend=dict(
            title='grouping',
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        width=1100,
        height=900
    )

    return fig


@st.cache(allow_output_mutation=True)
def prepare_figure(model, df):
    X = [text[:500] for text in df['text'].to_list()]
    filenames = df['filename'].to_list()

    X_transform, y = get_transform_and_predictions(model, X)

    fig = generate_plot(X_transform, y, filenames)

    return fig


@st.cache()
def prepare_page():
    model = load_model()
    df = load_dataset()

    X = [text[:500] for text in df['text'].to_list()]
    filenames = df['filename'].to_list()

    X_transform, y = get_transform_and_predictions(model, X)

    fig = prepare_figure(model, df)

    return fig, model


uploaded_files = st.sidebar.file_uploader("Upload your documents", accept_multiple_files=True,
                                          type=['pdf', 'txt'],
                                          help="Upload a set of .pdf or .txt files")

# button = st.sidebar.button('Organise Contracts', type='primary', use_container_width=True)

with st.spinner('⚙️ Loading model...'):
    fig, cuad_tfidf_umap_kmeans = prepare_page()
    figure = st.plotly_chart(fig, use_container_width=True)


if uploaded_files:
    figure.empty()
    filenames = []
    X_train = []
    if len(uploaded_files) < 5:
        st.error('### 💔 Please upload more than 4 files.')
    else:
        with st.spinner('⚙️ Training model...'):
            for uploaded_file in uploaded_files:
                print(uploaded_file.name)
                if '.pdf' in uploaded_file.name.lower():
                    reader = PdfReader(uploaded_file)
                    page_texts = [page.extract_text() for page in reader.pages]
                    text = "\n".join(page_texts)

                if '.txt' in uploaded_file.name.lower():
                    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                    text = stringio.read()

                X_train.append(text[:500])
                filenames.append(uploaded_file.name)

            if len(uploaded_files) < 10:
                n_clusters = 3
            else:
                n_clusters = 8

            tfidf_umap_kmeans = deepcopy(cuad_tfidf_umap_kmeans)
            tfidf_umap_kmeans.set_params(kmeans__n_clusters=n_clusters)
            tfidf_umap_kmeans.fit(X_train)

            X_transform, y = get_transform_and_predictions(cuad_tfidf_umap_kmeans, X_train)

        fig = generate_plot(X_transform, y, filenames)

        st.markdown("## 🗂 Your Organised Documents")

        st.plotly_chart(fig, use_container_width=True)


add_footer()

streamlit_analytics.stop_tracking(unsafe_password=os.environ["ANALYTICS_PASSWORD"])
