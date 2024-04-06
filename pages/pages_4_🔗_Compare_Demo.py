import os

import streamlit as st
import difflib
import spacy
import streamlit_analytics

from utils import add_logo_to_sidebar, add_footer


@st.cache(allow_output_mutation=True)
def load_model():
    return spacy.load('en_core_web_md')

streamlit_analytics.start_tracking()

## Layout stuff
st.set_page_config(
    page_title="Compare Demo",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:hello@simplexico.ai',
        'Report a bug': None,
        'About': "## This a demo showcasing different Legal AI Actions"
    }
)

add_logo_to_sidebar()

st.title('🔗 Compare Demo')
st.write("""This demo shows how AI can be used to compare passages of text.""")

with st.spinner('⚙️ Loading model...'):
    nlp = load_model()

EXAMPLE_TEXT_1 = """This Agreement shall be governed by and interpreted under the laws of the
State of Delaware without regard to its conflicts of law provisions."""

EXAMPLE_TEXT_2 = """This agreement will be governed by and must be construed in accordance with the laws of the State of Israel."""

col1, col2 = st.columns(2)
with col1:
    st.markdown('### 🖊 Enter a passage of text')
    text_1 = st.text_area('Enter a passage of text', label_visibility='collapsed', value=EXAMPLE_TEXT_1, height=100, key='input1')
with col2:
    st.markdown('### 🖊 Enter a second passage of text')
    text_2 = st.text_area('Enter a second passage of text', label_visibility='collapsed', value=EXAMPLE_TEXT_2, height=100, key='input2')

button = st.button('Compare', type='primary', use_container_width=True)


def get_tokens(doc):
    return [token.lower for token in doc]


def get_pos_tags(doc):
    return [token.pos_ for token in doc]


def add_md_color(text, match):
    color = 'green' if match else 'red'
    return f":{color}[{text}]"


def add_em(text, match):
    if match:
        return f"**{text}**"
    else:
        return f"*{text}*"


def create_str_output(doc, idxs):
    out = []
    for token in doc:
        text = token.text
        # higlight word diff
        if any(token.i in range(start, end) for start, end in idxs):
            text = add_md_color(text, match=True)
        else:
            text = add_md_color(text, match=False)

        out.append(text)

    return ' '.join(out)


def get_matching_idxs(items_1, items_2):
    sm = difflib.SequenceMatcher(None, items_1, items_2)
    matching_blocks = [match for match in sm.get_matching_blocks()]
    doc_1_matching_idxs = []
    doc_2_matching_idxs = []
    for a, b, n in matching_blocks:
        doc_1_matching_idxs.append((a, a + n))
        doc_2_matching_idxs.append((b, b + n))
    return doc_1_matching_idxs, doc_2_matching_idxs


if button:
    with st.spinner('⚙️ Comparing Texts...'):
        doc_1 = nlp(text_1)
        doc_2 = nlp(text_2)

    st.header('🧪 Comparison')
    st.markdown('We can highlight the :green[similarities] and :red[differences] in **wording** across the two texts.')

    doc_1_token_idxs, doc_2_token_idxs = get_matching_idxs(get_tokens(doc_1), get_tokens(doc_2))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(create_str_output(doc_1, doc_1_token_idxs))
    with col2:
        st.markdown(create_str_output(doc_2, doc_2_token_idxs))

    col1, col2, col3 = st.columns(3)

    with col1:
        # perform simple sequence matching
        sm = difflib.SequenceMatcher(None, get_tokens(doc_1), get_tokens(doc_2))
        st.subheader('📑 Textual Similarity')
        st.markdown('We can measure the similarity based on the *wording* of the two texts.')
        st.metric(label='Textual Similarity', value=f"{sm.ratio() * 100:.1f}%")

    with col2:
        st.subheader('📏 Linguistic Similarity')
        st.markdown('We can measure the similarity based on the *linguistic features* of the two texts.')
        postags_1 = [token.pos_ for token in doc_1]
        postags_2 = [token.pos_ for token in doc_2]
        sm = difflib.SequenceMatcher(None, postags_1, postags_2)
        st.metric(label='Linguistic Similarity', value=f"{sm.ratio() * 100:.1f}%")

    with col3:
        st.subheader('💭 Semantic Similarity')
        st.markdown('We can measure the similarity based on the *meaning* of the two texts.')
        st.metric(label='Semantic Similarity', value=f"{doc_1.similarity(doc_2) * 100:.1f}%")

    add_footer()

streamlit_analytics.stop_tracking(unsafe_password=os.environ["ANALYTICS_PASSWORD"])