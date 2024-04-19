import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Define function to save feedback to Google Sheets
def save_feedback_to_google_sheets(feedback):
    # Load credentials from the downloaded JSON key file
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = ServiceAccountCredentials.from_json_keyfile_name("feedback-db-420816-d91b0eb69252.json", scope)
    
    # Authenticate with Google Sheets API
    gc = gspread.authorize(credentials)
    
    # Open the Google Sheets spreadsheet by its URL
    sheet_url = "https://docs.google.com/spreadsheets/d/1QFg7ZEec0O1-O0Sl8MSBr9biLerqdz1TrGtAEpLCfcc/edit?usp=sharing"
    sh = gc.open_by_url(sheet_url)
    
    # Select the first (default) worksheet
    worksheet = sh.get_worksheet(0)
    
    # Append feedback to the Google Sheets spreadsheet
    row = [feedback]
    worksheet.append_row(row)
    st.success("Feedback saved to Google Sheets successfully!")

def main():
    st.title("Feedback Form")
    
    with st.form("main", clear_on_submit=True):
        st.write('Click any ⬇️')
        
        feedback = st.text_input("Please write your feedback (optional)")
        
        if st.form_submit_button("Submit Feedback"):
            save_feedback_to_google_sheets(feedback)

if __name__ == "__main__":
    main()
