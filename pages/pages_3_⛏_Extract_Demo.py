import os
import re

import streamlit_analytics
from huggingface_hub import snapshot_download

import streamlit as st
import streamlit.components.v1 as components

import spacy
from spacy import displacy
from spacy.tokens import Span

import pandas as pd
import numpy as np

from utils import add_logo_to_sidebar, add_footer


HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "simplexico/cuad-spacy-custom-ner"

EXAMPLE_TEXT = """Exhibit 10.16 CONSULTING AGREEMENT 
This Consulting Agreement (the "Agreement") is made and entered into as of this 2nd day of January 2020, 
by and between Global Technologies, Ltd (hereinafter the "Company"), 
a Delaware corporation whose address is 501 1st Ave N., Suite 901, St. Petersburg, FL 33701 and Timothy Cabrera (hereinafter the "Consultant"), 
an individual whose address is 11718 SE Federal Hwy., Suite 372, Hobe Sound, FL 33455 (individually, a "Party"; collectively, the "Parties")."""

streamlit_analytics.start_tracking()

## Layout stuff
st.set_page_config(
    page_title="Extract Demo",
    page_icon="⛏",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:hello@simplexico.ai',
        'Report a bug': None,
        'About': "## This a demo showcasing different Legal AI Actions"
    }
)

add_logo_to_sidebar()

st.title('⛏ Extract Demo')
st.write("""
This demo shows how AI can be used to extract information from text.
We've trained an AI model to extract key pieces of information from a contract recital.
""")

@st.cache(allow_output_mutation=True)
def load_model():
    snapshot_download(repo_id=REPO_ID, token=HF_TOKEN, local_dir='./')
    nlp = spacy.load('model-best')
    return nlp

st.markdown('### 🖊 Enter a contract recital')
text = st.text_area('Enter Clause Text', label_visibility='collapsed', value=EXAMPLE_TEXT, height=100)
button = st.button('Extract Data', type='primary', use_container_width=True)

with st.spinner('⚙️ Loading model...'):
    nlp = load_model()


def check_span_pair_for_overlap(span1, span2):
    """ Checks a pair of spans for any overlapping ranges
    Args:
        span1: (start, end) tuple
        span2: (start, end) tuple
    Return:
        True if overlap, False otherwise
    """
    # remove offset
    minimum = min(span1[0], span2[0])
    span1 = (span1[0] - minimum, span1[1] - minimum)
    span2 = (span2[0] - minimum, span2[1] - minimum)

    maximum = max(span1[1], span2[1])
    vec1 = np.zeros(maximum)
    vec1[span1[0]:span1[1]] = 1
    vec2 = np.zeros(maximum)
    vec2[span2[0]:span2[1]] = 1
    if np.dot(vec1, vec2):
        return True
    return False


def add_detected_persons_as_parties(doc):
    nlp = spacy.load('en_core_web_md')

    doc_ = nlp(doc.text)
    original_ents = list(doc.ents)

    for ent in doc_.ents:
        if ent.label_ == 'PERSON':
            if not any([check_span_pair_for_overlap((ent.start, ent.end), (ent_.start, ent_.end)) for ent_ in
                        original_ents]):
                print(ent)
                # check for overlapping ents

                original_ents.append(Span(doc, ent.start, ent.end, label='parties'))

    doc.ents = original_ents

    return doc


def add_rule_based_entites(doc):
    """Adds rule based entity spans to document
    Args:
        doc (spacy.tokens.doc.Doc)
    """
    patterns = [
        ('[0-9]+[\s]+[a-zA-Z0-9.\-\,\#]+[\s]*[a-zA-Z0-9.\-\,\#]+[a-zA-Z0-9\s.\-\,\#]*\s[0-9]+', 'address'),
        ('Consultant|Company|Party|Parties', 'role'),
    ]

    for pattern, label in patterns:
        ents = []
        for match in re.finditer(pattern, doc.text):
            start, end = match.span()
            span = doc.char_span(start, end)
            if span is not None:
                ents.append((span.start, span.end, span.text))

        original_ents = list(doc.ents)

        for ent in ents:
            start, end, address = ent
            per_ent = Span(doc, start, end, label=label)
            original_ents.append(per_ent)

        doc.ents = original_ents

    return doc


if button:
    col1, col2 = st.columns(2)

    with st.spinner('⚙️ Extracting Data...'):
        doc = nlp(text)
        doc = add_rule_based_entites(doc)
        doc = add_detected_persons_as_parties(doc)

    with col1:
        st.subheader('🎨 Highlighted Text')

        colors = {'party': "#85C1E9", "address": "#ff6961", "agreement_date": "#5de36f", "role": "#b05de3"}
        options = {"ents": ['party', 'address', 'agreement_date', 'role'], "colors": colors}

        label_aliases = {
            'parties': 'Party',
            'address': 'Address',
            'agreement_date': 'Agreement Date',
            'role': 'Role'
        }

        doc.spans["sc"] = [
            Span(doc, ent.start, ent.end, label_aliases[ent.label_]) for ent in doc.ents
        ]

        html = displacy.render(doc, style="span", options=options)
        components.html(html, height=400)

    with col2:
        # display table
        data = {
            'Text': [],
            'Label': []
        }

        st.subheader('📊 Extracted Data')
        for span in doc.spans['sc']:
            data['Label'].append(span.label_)
            data['Text'].append(span.text)
        df = pd.DataFrame(data)

        hide_dataframe_row_index = """
                <style>
                .row_heading.level0 {display:none}
                .blank {display:none}
                </style>
                """

        # Inject CSS with Markdown
        st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

        st.table(df)

    add_footer()

streamlit_analytics.stop_tracking(unsafe_password=os.environ["ANALYTICS_PASSWORD"])