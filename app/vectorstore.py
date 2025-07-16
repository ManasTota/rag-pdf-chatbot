# app/vectorstore.py
import os
from langchain_community.vectorstores import FAISS
# NEW: For Google AI Studio Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from utils import AI_STUDIO_EMBEDDING_MODEL, GOOGLE_API_KEY  # Import config from utils


class VectorStoreManager:
    def __init__(self):
        # Set the API key as environment variable if not already set
        if GOOGLE_API_KEY and not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

        # Initialize Google Generative AI embeddings model
        # The GoogleGenerativeAIEmbeddings class picks up the API key from environment automatically
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=AI_STUDIO_EMBEDDING_MODEL
            # Remove the google_api_key parameter - it's not supported in the constructor
        )
        self.vectorstore = None
        self.db_path = "faiss_index"  # Directory to save the FAISS index

    def create_and_save_vectorstore(self, chunks: list[Document], db_name: str = "default_index"):
        """
        Creates a FAISS vector store from document chunks and saves it locally.
        """
        if not chunks:
            raise ValueError("No chunks provided to create vector store.")

        print(
            f"Creating FAISS vector store with {len(chunks)} chunks using {AI_STUDIO_EMBEDDING_MODEL}...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # Ensure the directory exists
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

        save_path = os.path.join(self.db_path, db_name)
        self.vectorstore.save_local(save_path)
        print(f"FAISS index saved to {save_path}")
        return self.vectorstore

    def load_vectorstore(self, db_name: str = "default_index"):
        """
        Loads an existing FAISS vector store from local storage.
        """
        load_path = os.path.join(self.db_path, db_name)
        if not os.path.exists(load_path):
            print(
                f"FAISS index not found at {load_path}. A new one might be created or you need to process a PDF first.")
            return None

        print(f"Loading FAISS vector store from {load_path}...")
        self.vectorstore = FAISS.load_local(
            load_path, self.embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded.")
        return self.vectorstore

    def get_retriever(self, k: int = 4):
        """
        Returns a retriever object from the loaded vector store.
        """
        if self.vectorstore is None:
            raise ValueError(
                "Vector store not initialized. Call create_and_save_vectorstore or load_vectorstore first.")

        return self.vectorstore.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    # Test vector store manager independently
    from app.utils import load_and_chunk_pdf  # Import for full test

    # For standalone test, ensure GOOGLE_API_KEY is set in your actual env or for this run
    if not GOOGLE_API_KEY:
        print("Warning: GOOGLE_API_KEY not set for standalone vectorstore test.")
        print("Please ensure it's in your .env or set it manually for this test.")
        # Create a dummy API key for test if not present to avoid immediate error
        # In real scenario, this would fail if GOOGLE_API_KEY is truly missing.
        os.environ["GOOGLE_API_KEY"] = "dummy_key_for_test"

    dummy_pdf_path = "../data/example.pdf"
    try:
        # Create a dummy PDF if it doesn't exist for testing purposes
        if not os.path.exists("../data"):
            os.makedirs("../data")
        if not os.path.exists(dummy_pdf_path):
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(dummy_pdf_path)
            c.drawString(100, 750, "This is a test PDF for vectorstore.py.")
            c.save()

        test_chunks = load_and_chunk_pdf(dummy_pdf_path)
        manager = VectorStoreManager()
        created_vs = manager.create_and_save_vectorstore(
            test_chunks, "test_ai_studio")
        print(f"Created vectorstore: {created_vs}")

        loaded_vs = manager.load_vectorstore("test_ai_studio")
        print(f"Loaded vectorstore: {loaded_vs}")

        if loaded_vs:
            manager.vectorstore = loaded_vs  # Update manager's internal state
            retriever = manager.get_retriever()
            results = retriever.invoke("What is a test PDF?")
            print("Retrieved docs:")
            for doc in results:
                print(f"- {doc.page_content[:100]}...")

        # Clean up
        import shutil
        if os.path.exists(manager.db_path):
            shutil.rmtree(manager.db_path)
        if os.path.exists(dummy_pdf_path):
            os.remove(dummy_pdf_path)

    except Exception as e:
        print(f"Error during vectorstore test: {e}")
