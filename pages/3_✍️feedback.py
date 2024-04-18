import streamlit as st
from streamlit_feedback import streamlit_feedback
st.header("Feedback Form")
with st.form("main", clear_on_submit=True):
    st.write('write 👇 ...')
   
    feedback = streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="[Optional] Please provide an explanation",
        align="flex-start"
    )
    st.form_submit_button('Submit Feedback')


    # Check if the form was submitted
    if submit_button:
        # Return the feedback text upon form submission
        return feedback_text

    # If the form was not submitted, return None
    return None
st.write(f"feedback log -{feedback}")
