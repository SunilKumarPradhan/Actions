import os
from io import StringIO
import re

import pandas as pd

import streamlit as st
import streamlit_analytics

import streamlit_toggle as tog
from pypdf import PdfReader

from utils import add_logo_to_sidebar, add_footer, add_email_signup_form

from huggingface_hub import snapshot_download

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, EmbeddingRetriever

HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_REPO_ID = "simplexico/cuad-qa-answers"
DATA_FILENAME = "cuad_questions_answers.json"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
if EMBEDDING_MODEL == "sentence-transformers/multi-qa-MiniLM-L6-cos-v1" or EMBEDDING_MODEL == "sentence-transformers/paraphrase-MiniLM-L3-v2":
    EMBEDDING_DIM = 384
else:
    EMBEDDING_DIM = 768

EXAMPLE_TEXT = "the governing law is the State of Texas"

streamlit_analytics.start_tracking()


@st.cache(allow_output_mutation=True)
def load_dataset():
    snapshot_download(repo_id=DATA_REPO_ID, token=HF_TOKEN, local_dir='./', repo_type='dataset')
    df = pd.read_json(DATA_FILENAME)
    return df


@st.cache(allow_output_mutation=True)
def generate_document_store(df):
    """Create haystack document store using contract clause data 
    """
    document_dicts = []

    for idx, row in df.iterrows():
        document_dicts.append(
            {
                'content': row['paragraph'],
                'meta': {'contract_title': row['contract_title']}
            }
        )

    document_store = InMemoryDocumentStore(use_bm25=True, embedding_dim=EMBEDDING_DIM, similarity='cosine')

    document_store.write_documents(document_dicts)

    return document_store


def files_to_dataframe(uploaded_files, limit=10):
    texts = []
    titles = []
    for uploaded_file in uploaded_files[:limit]:
        if '.pdf' in uploaded_file.name.lower():
            reader = PdfReader(uploaded_file)
            page_texts = [page.extract_text() for page in reader.pages]
            text = "\n".join(page_texts).strip()

        if '.txt' in uploaded_file.name.lower():
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            text = stringio.read().strip()

        paragraphs = text.split("\n")
        paragraphs = [p.strip() for p in paragraphs if len(p.split()) > 10]
        texts.extend(paragraphs)
        titles.extend([uploaded_file.name] * len(paragraphs))

    return pd.DataFrame({'paragraph': texts, 'contract_title': titles})


@st.cache(allow_output_mutation=True)
def generate_bm25_retriever(document_store):
    return BM25Retriever(document_store)


@st.cache(allow_output_mutation=True)
def generate_embeddings(embedding_model, document_store):
    embedding_retriever = EmbeddingRetriever(
        embedding_model=embedding_model,
        document_store=document_store,
        model_format="sentence_transformers",
        scale_score=True
    )
    document_store.update_embeddings(embedding_retriever)
    return embedding_retriever


def process_query(query, retriever):
    """Generates dataframe with top ten results"""
    texts = []
    contract_titles = []
    scores = []
    ranking = []
    candidate_documents = retriever.retrieve(
        query=query,
        top_k=10,
    )

    for idx, document in enumerate(candidate_documents):
        texts.append(document.content)
        contract_titles.append(document.meta["contract_title"])
        scores.append(str(round(document.score, 2)))
        ranking.append(idx + 1)

    return pd.DataFrame(
        {
            "Rank": ranking,
            "Text": texts,
            "Source Document": contract_titles,
            "Similarity Score": scores
        }
    )


st.set_page_config(
    page_title="Find Demo",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:hello@simplexico.ai',
        'Report a bug': None,
        'About': "## This a demo showcasing different Legal AI Actions"
    }
)

add_logo_to_sidebar()

st.title('🔎 Find Demo')

st.write("""
This demo shows how a set of documents can be searched.
Upload a set of documents on the left and the paragraphs can be searched using **keyword** or using **semantic** search.
Semantic search leverages an AI model which matches on paragraphs with a similar meaning to the input text. 
""")

st.info("**👈 Upload a set of documents on the left**")

uploaded_files = st.sidebar.file_uploader("Upload a set of documents **(upload up to 10 files)**",
                                          type=['pdf', 'txt'],
                                          help='Upload a set of .pdf or .txt files',
                                          accept_multiple_files=True)

if uploaded_files:
    with st.spinner('🔺 Uploading files...'):
        df = files_to_dataframe(uploaded_files)
        document_store = generate_document_store(df)

    st.write("**👇 Enter a search query below** and toggle keyword/semantic mode and hit **Search**")
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(label='Enter Search Query', label_visibility='collapsed', value=EXAMPLE_TEXT)
    with col2:
        value = tog.st_toggle_switch(
            label="Semantic Mode",
            label_after=False,
            inactive_color='#D3D3D3',
            active_color="#11567f",
            track_color="#29B5E8"
        )
        if value:
            search_type = "semantic"
        else:
            search_type = "keyword"

    button = st.button('Search', type='primary', use_container_width=True)

    if button:

        hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """

        st.subheader(f'✅ {search_type.capitalize()} Search Results')
        # Inject CSS with Markdown
        st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

        if search_type == "keyword":
            with st.spinner('⚙️ Running search...'):
                bm25_retriever = generate_bm25_retriever(document_store)
                df_bm25 = process_query(query, bm25_retriever)
            st.table(df_bm25)

        if search_type == "semantic":
            with st.spinner('⚙️ Running search...'):
                embedding_retriever = generate_embeddings(EMBEDDING_MODEL, document_store)
                df_embed = process_query(query, embedding_retriever)
            st.table(df_embed)

        add_footer()

streamlit_analytics.stop_tracking(unsafe_password=os.environ["ANALYTICS_PASSWORD"])
