import streamlit as st
from rag import ask_question

st.set_page_config(page_title="Zero-Cost AI Chatbot", page_icon="🤖")

st.title("📚 Zero-Cost AI Chatbot")
st.write("Ask questions about your documents and get answers with sources.")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("You:", "")

if user_input:
    result = ask_question(user_input)
    st.session_state.history.append({"user": user_input, "ai": result["answer"], "sources": result["sources"]})

# Display chat history
for chat in st.session_state.history[::-1]:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**AI:** {chat['ai']}")
    if chat['sources']:
        sources = ", ".join([f"{s['source']} (page {s.get('page','?')})" for s in chat['sources']])
        st.markdown(f"**Sources:** {sources}")
    st.markdown("---")
