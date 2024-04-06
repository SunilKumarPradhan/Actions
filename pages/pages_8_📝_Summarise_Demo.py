import os
import re

import streamlit as st
import streamlit_analytics
from utils import add_logo_to_sidebar, add_footer, add_email_signup_form

from transformers import pipeline

SUMMARIZER_MODEL = 't5-small'

EXAMPLE_TEXT = """Each Party will notify the other Party in writing in the event it becomes aware of a claim for which indemnification may be sought hereunder. 
In the event that any Third Party asserts a claim or other proceeding (including any governmental investigation) with respect to any matter for which a Party (the "Indemnified Party") is entitled to indemnification hereunder (a "Third Party Claim"),
then the Indemnified Party shall promptly notify the Party obligated to indemnify the Indemnified Party (the "Indemnifying Party") thereof;
provided, however, that no delay on the part of the Indemnified Party in notifying the Indemnifying Party shall relieve the Indemnifying Party from any obligation hereunder unless (and then only to the extent that) the Indemnifying Party is prejudiced thereby."""


@st.cache(allow_output_mutation=True)
def load_summarizer():
    return pipeline("summarization", model=SUMMARIZER_MODEL)


def clean_text(text):
    """Reformat summarizer output"""
    text = re.sub(r'\s\.', '.', text)
    text = '. '.join([sent.capitalize() for sent in text.split('. ')])
    return text


streamlit_analytics.start_tracking()

st.set_page_config(
    page_title="Summarise Demo",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:hello@simplexico.ai',
        'Report a bug': None,
        'About': "## This a demo showcasing different Legal AI Actions"
    }
)

add_logo_to_sidebar()

st.title('📝 Summarise Demo')
st.write('We can use AI to summarise the text of a paragraph, maintaining the most pertinent information in the paragraph. Enter a clause below and click _summarise_ to see the automatic summarisation')

st.markdown('### 🖊 Enter a Clause')
text = st.text_area('Enter Clause Text', label_visibility='collapsed', value=EXAMPLE_TEXT.replace('\n', ' '),
                    height=100)
button = st.button('Summarise', type='primary', use_container_width=True)

if button:

    with st.spinner('⚙️ Summarising Clause...'):
        prefix = "summarize: "
        summarizer = load_summarizer()
        summarized_text = summarizer(prefix + text)[0]['summary_text']
        summarized_text = clean_text(summarized_text)
    st.markdown('### Summarised Clause:')

    st.markdown(f":gray[{summarized_text}]")

st.write("---")

add_email_signup_form()

add_footer()

streamlit_analytics.stop_tracking(unsafe_password=os.environ["ANALYTICS_PASSWORD"])
