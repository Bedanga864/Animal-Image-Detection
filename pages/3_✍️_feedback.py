import streamlit as st
from streamlit_feedback import streamlit_feedback

st.set_page_config(
  page_title="Animal Image Detection App",
  page_icon="📱",
)

st.header("Feedback Form")

with st.form("main", clear_on_submit=True):
    st.write('click any ⬇️')
   
    feedback = streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="[Optional] Please provide an explanation",
        align="flex-start"
    )
    st.form_submit_button('save')

st.write(f"feedback log -{feedback}")
