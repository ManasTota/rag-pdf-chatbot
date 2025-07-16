# app/chat_logic.py
from langchain_google_genai import ChatGoogleGenerativeAI # NEW: Google AI Studio Chat Model
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from typing import List
import os

from utils import AI_STUDIO_LLM_MODEL, GOOGLE_API_KEY # Import config from utils

class ChatbotLogic:
    def __init__(self, retriever):
        # Set the API key as environment variable if not already set
        if GOOGLE_API_KEY and not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
            
        # Initialize Google Generative AI LLM model
        self.llm = ChatGoogleGenerativeAI(
            model=AI_STUDIO_LLM_MODEL,
            temperature=0.0 # Keep temperature low for factual Q&A
            # Removed convert_system_message_to_human as it's deprecated
            # Removed google_api_key parameter - it's picked up from environment
        )
        self.retriever = retriever

        # 1. Prompt for the LLM
        self.qa_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Keep the answer concise and accurate.\n\n"
            "{context}"
        )
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.qa_system_prompt),
            ("human", "{input}")
        ])

        # 2. Document combining chain
        self.document_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)

        # 3. Retrieval chain
        self.retrieval_qa_chain = create_retrieval_chain(self.retriever, self.document_chain)

    def ask_question(self, question: str):
        """
        Asks a question to the RAG chatbot.
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized. Ensure vector store is loaded.")

        print(f"\n--- Answering Question: '{question}' ---")
        try:
            response = self.retrieval_qa_chain.invoke({"input": question})
            
            answer = response.get("answer")
            source_documents = response.get("context", [])

            print(f"AI Answer: {answer}")
            if source_documents:
                print("\nSources (from retrieved documents):")
                for i, doc in enumerate(source_documents):
                    page_info = f"Page {doc.metadata.get('page', 'N/A')}" if doc.metadata else "Page N/A"
                    print(f"  - Document {i+1} ({page_info}): {doc.page_content[:150]}...")
            else:
                print("No source documents found for this query.")

            return answer, source_documents
        
        except Exception as e:
            print(f"Error during question processing: {e}")
            raise e

if __name__ == "__main__":
    # This is for testing the chat logic independently
    # Mocking a retriever that returns dummy documents
    class MockRetriever:
        def invoke(self, query: str) -> List[Document]:
            if "google" in query.lower():
                return [Document(page_content="Google Generative AI provides powerful models like Gemini.", metadata={"page": 1})]
            elif "llm" in query.lower():
                return [Document(page_content="Large Language Models (LLMs) are powerful AI models.", metadata={"page": 5})]
            else:
                return [Document(page_content="General knowledge about various topics.", metadata={"page": 1})]

    print("--- Testing ChatbotLogic with MockRetriever ---")
    mock_retriever = MockRetriever()
    chatbot = ChatbotLogic(retriever=mock_retriever)

    answer1, sources1 = chatbot.ask_question("What is Google Generative AI?")
    print(f"Answer: {answer1}")

    answer2, sources2 = chatbot.ask_question("Tell me about LLMs.")
    print(f"Answer: {answer2}")

    answer3, sources3 = chatbot.ask_question("What is the capital of France?")
    print(f"Answer: {answer3}")