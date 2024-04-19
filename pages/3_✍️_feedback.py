import streamlit as st
from streamlit_feedback import streamlit_feedback

st.set_page_config(
  page_title="Animal Image Detection App",
  page_icon="📱",
)

def save_feedback_to_file(feedback):
    # Define the file path where feedback logs will be stored
    file_path = "feedback_logs.txt"

st.header("Feedback")

with st.form("main", clear_on_submit=True):
    st.write('click any ⬇️')
   
    feedback = streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="[Optional] Please write your feedback",
        align="flex-start"
    )
    st.form_submit_button('save')

if st.button("Submit Feedback"):
        save_feedback_to_file(email, feedback_text, rating)
        st.success("Feedback submitted successfully!")

st.write(f"feedback log -{feedback}")
