import streamlit as st
from streamlit_feedback import streamlit_feedback
def show_feedback_form():
    st.header("Feedback Form")
    
    # Add input fields for the feedback form
    feedback_text = st.text_area("Please enter your feedback here:")
    
    # Add a button to submit feedback
    if st.button("Submit Feedback"):
        # Here you can process the feedback (e.g., save to a file or database)
        # For simplicity, let's just display a confirmation message
        st.success("Thank you for your valuable feedback!")
def main():
    
    # Display the feedback form
    show_feedback_form()

if __name__ == "__main__":
    main()
