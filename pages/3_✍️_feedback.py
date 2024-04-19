import streamlit as st
from streamlit_feedback import streamlit_feedback

with st.form("main", clear_on_submit=True):
    st.write('answer ...')
   
    feedback = streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="[Optional] Please provide an explanation",
        align="flex-start"
    )
st.write(f"feedback log -{feedback}")
