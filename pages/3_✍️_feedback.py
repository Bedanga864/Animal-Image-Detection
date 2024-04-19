import streamlit as st
from streamlit_feedback import streamlit_feedback
st.set_page_config(
  page_title="Animal Image Detection App",
  page_icon="📱",
)
st.header("Feedback Form")
with st.form("main", clear_on_submit=True):
    st.write('Click any ⬇️')
   
    feedback = streamlit_feedback(
       # feedback_type="five stars",
        rating = st.slider("Rate your experience (1 - 5 stars)", min_value=1, max_value=5, step=1)
        optional_text_label="[Optional] Please write your feedback",
        align="flex-start"
    )
    st.form_submit_button('save')
st.write(f"feedback log -{feedback}")
