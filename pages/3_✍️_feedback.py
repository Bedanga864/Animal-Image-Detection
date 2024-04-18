import streamlit as st
import smtplib
from email.message import EmailMessage
def show_feedback_form():
    st.header("Feedback Form")

    # Add input fields for the feedback form
    email = st.text_input("Your Email (optional):")
    feedback_text = st.text_area("Please enter your feedback here:")

    # Add a button to submit feedback
    if st.button("Submit Feedback"):
        send_feedback(email, feedback_text)
        st.success("Feedback submitted successfully!")
def send_feedback(email, feedback_text):
    # Set up SMTP server details
    smtp_server = "your_smtp_server_address"
    smtp_port = 587  # Change to the appropriate port for your SMTP server
    sender_email = "your_sender_email@example.com"
    password = "your_email_password"  # Use an app-specific password if available

    # Create EmailMessage object
    message = EmailMessage()
    message["From"] = sender_email
    message["To"] = "recipient_email@example.com"  # Change to your recipient's email address
    message["Subject"] = "Feedback from Streamlit App"
    message.set_content(f"Feedback from {email}:\n\n{feedback_text}")

    # Connect to SMTP server and send email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, password)
        server.send_message(message)
def main():
    # Display the feedback form
    show_feedback_form()

if __name__ == "__main__":
    main()

