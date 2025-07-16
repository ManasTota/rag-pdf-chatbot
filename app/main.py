# app/main.py
import streamlit as st
import os
import asyncio
import sys

# Imports from your app modules
from utils import load_and_chunk_pdf
from vectorstore import VectorStoreManager
from chat_logic import ChatbotLogic


if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


# --- Initialize Session State ---
if "vector_store_manager" not in st.session_state:
    st.session_state.vector_store_manager = VectorStoreManager()
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "messages" not in st.session_state:
    st.session_state.messages = []  # For chat history

# --- Streamlit UI ---
st.set_page_config(page_title="RAG PDF Chatbot (Google AI Studio)",
                   page_icon="📄", layout="centered")
st.title("📄 RAG PDF Chatbot (Google AI Studio)")
st.markdown("Upload a PDF and chat with its content using Google AI Studio!")

# 1. PDF Upload Section
st.header("1. Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file and not st.session_state.pdf_processed:
    with st.spinner("Processing PDF... This might take a moment."):
        # Save the uploaded file temporarily
        os.makedirs("temp_pdfs", exist_ok=True)
        temp_pdf_path = os.path.join("temp_pdfs", uploaded_file.name)
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            # Load and chunk the PDF
            chunks = load_and_chunk_pdf(temp_pdf_path)
            st.success(f"PDF processed! Found {len(chunks)} text chunks.")

            # Create and save the vector store
            db_name = os.path.splitext(uploaded_file.name)[0]
            st.session_state.vector_store_manager.create_and_save_vectorstore(
                chunks, db_name=db_name)

            # Initialize the chatbot with the retriever from the new vector store
            retriever = st.session_state.vector_store_manager.get_retriever()
            st.session_state.chatbot = ChatbotLogic(retriever=retriever)
            st.session_state.pdf_processed = True
            st.success("PDF successfully loaded and ready for chat!")

            # Clean up the temporary PDF file
            os.remove(temp_pdf_path)

        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            st.session_state.pdf_processed = False  # Reset if error
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)


# 2. Chat Interface Section
st.header("2. Chat with PDF")

if st.session_state.pdf_processed and st.session_state.chatbot:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about the PDF..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Ask the question using our chatbot logic
                    answer, source_documents = st.session_state.chatbot.ask_question(
                        prompt)

                    full_response = answer
                    if source_documents:
                        sources_text = "\n\n**Sources:**\n"
                        for i, doc in enumerate(source_documents):
                            page_info = f"Page {doc.metadata.get('page', 'N/A')}" if doc.metadata else "Page N/A"
                            sources_text += f"- *{doc.page_content[:100]}...* ({page_info})\n"
                        full_response += sources_text

                    st.markdown(full_response)
                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Error getting answer: {e}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Sorry, an error occurred: {e}"})
else:
    st.info("Please upload a PDF to start chatting.")
