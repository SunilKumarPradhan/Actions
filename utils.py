import base64
import os
from datetime import datetime
from pathlib import Path

import mailerlite as MailerLite
import streamlit as st
import streamlit.components.v1 as components


client = MailerLite.Client({
    'api_key': os.environ['mailerlitetoken']
})
NEWSLETTER_GROUP_ID = int(os.environ['NEWSLETTERGROUPID'])


def get_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def add_logo_to_sidebar():
    st.markdown(
        f"""
            <style>
                [data-testid="stSidebarNav"] {{
                    background-image: url(data:image/png;base64,{base64.b64encode(Path('logo.png').read_bytes()).decode()});
                    background-repeat: no-repeat;
                    background-position: 20px 20px;
                    background-size: 300px;
                    padding-top: 100px
                }}
            </style>
            """,
        unsafe_allow_html=True,
    )


def add_share_to_twitter_button():
    return components.html(
        """
            <a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button" 
            data-text="Checkout the Legal AI Actions Demo from @simplexico_ 🎈" 
            data-url="https://simplexico-legal-ai-actions.hf.space"
            data-show-count="false">
            data-size="Large" 
            data-hashtags="legalai,legalnlp"
            Tweet
            </a>
            <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
        """, height=30
    )


def add_footer():
    st.info("""
    ### 🙋‍♂️ Interested in building out your own tailored Legal AI solutions?
    - 🌐 Check out our [website](https://simplexico.ai)
    - 📞 Book a call with [us](https://calendly.com/uwais-iqbal/discovery-call)
    - ✉️ Send us an [email](mailto:hello@simplexico.ai)
    """)

    st.success("""
    #### 🙌 Follow Us on Social Media - [🐥 Twitter](https://twitter.com/_simplexico) | [💼 LinkedIn](https://www.linkedin.com/company/simplexico/?viewAsMember=true)
    """)


def add_email_signup_form():
    st.markdown("### 💌 Join our mailing list!")
    st.markdown('Keep up to date with all things simplexico with our monthly newsletter.')
    col1, col2 = st.columns(2)
    with st.form(key='email-form'):
        name = col1.text_input(label='Enter your name', placeholder='John Doe')
        email = col2.text_input(label='Enter your email', placeholder='john.doe@outlook.com')

        submit_button = st.form_submit_button(label='Submit', type='primary', use_container_width=True)

        if submit_button:
            valid_name = True
            valid_email = True

            if name == "":
                st.error('❌ Error! Please enter a name.')
                valid_name = False

            if email == "":
                st.error('❌ Error! Please enter an email.')
                valid_email = False
            elif not '@' in email:
                st.error('❌ Error! Please enter a valid email.')
                valid_email = False
            elif not '.' in email.split('@')[-1]:
                st.error('❌ Error! Please enter a valid email.')
                valid_email = False

            if valid_name and valid_email:
                response = client.subscribers.create(email, fields={'name': name},
                                                     groups=[NEWSLETTER_GROUP_ID],
                                                     status='active', subscribed_at=get_timestamp())
                try:
                    if response['data']['status'] == 'active':
                        st.success(f'✅ 👋 Hey {name}! Welcome to our mailing list.')
                except Exception as e:
                    st.error(f"😕 Sorry {name}. Something went wrong. We weren't able to add you to our mailing list.")
