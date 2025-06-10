import streamlit as st

st.set_page_config(page_title="Layout Test")

st.header("Layout Test")
st.info("This is a simple test to check button placement.")
st.write("The 'Clear Chat' button should appear at the very bottom, just above the text input bar.")

# Simulate some chat history so the page has content
for i in range(3):
    with st.chat_message("assistant"):
        st.write(f"This is a sample message {i+1}")

st.markdown("---")
st.write("This is the end of the page's main content.")
st.markdown("---")


# This is the chat input widget that should be stuck to the bottom of the screen.
user_input = st.chat_input("Your input here...")

# This button is defined AFTER the chat input in the script.
# It should appear in the page flow, which means it will be the
# LAST thing before the input bar.
if st.button("Clear Chat"):
    st.balloons()