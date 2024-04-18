import streamlit as st
from streamlit_feedback import streamlit_feedback
    
def show_feedback_form():
    st.header("Feedback Form")

    # Text area for feedback input
    feedback_text = st.text_area("Please enter your feedback here:")

    # Submit button
    if st.button("Submit Feedback"):
        return feedback_text

    return None
def save_feedback(feedback_text):
    if feedback_text:
        with open("feedback.txt", "a") as file:
            file.write(feedback_text + "\n")
            st.success("Feedback submitted successfully!")
    else:
        st.warning("Please enter your feedback before submitting.")
def main():

    # Show the feedback form
    feedback_text = show_feedback_form()

    # If feedback is submitted, save it to file
    if feedback_text is not None:
        save_feedback(feedback_text)

if __name__ == "__main__":
    main()

