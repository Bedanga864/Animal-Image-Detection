import streamlit as st

def main():

    # Paste your JotForm embed code here
    jotform_embed_code = """
    
    <iframe
      id="JotFormIFrame-241112385506449"
      title="Feedback Form"
      onload="window.parent.scrollTo(0,0)"
      allowtransparency="true"
      allow="geolocation; microphone; camera; fullscreen"
      src="https://form.jotform.com/241112385506449"
      frameborder="0"
      style="min-width:100%;max-width:100%;height:539px;border:none;"
      scrolling="no"
    >
    
    </iframe>
    """

    # Display the JotForm using HTML iframe
    st.markdown(jotform_embed_code, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
