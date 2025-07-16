# app/utils.py
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google AI Studio API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Updated model names for current Google AI Studio API
AI_STUDIO_LLM_MODEL = os.getenv("AI_STUDIO_LLM_MODEL", "gemini-1.5-flash")  # Changed from gemini-pro
AI_STUDIO_EMBEDDING_MODEL = os.getenv("AI_STUDIO_EMBEDDING_MODEL", "models/embedding-001")  # Added models/ prefix

# Debug: Print the actual values being used
print(f"DEBUG: Using LLM model: {AI_STUDIO_LLM_MODEL}")
print(f"DEBUG: Using embedding model: {AI_STUDIO_EMBEDDING_MODEL}")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found in .env file. Please get your API key from Google AI Studio and add it.")

# --- PDF Loading and Chunking ---

def load_and_chunk_pdf(pdf_path: str) -> list[Document]:
    """
    Loads a PDF document and splits it into chunks.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    # Test loading env vars
    print(f"Google API Key loaded (first 5 chars): {GOOGLE_API_KEY[:5]}...")
    print(f"LLM Model from .env: {AI_STUDIO_LLM_MODEL}")
    print(f"Embedding Model from .env: {AI_STUDIO_EMBEDDING_MODEL}")

    # Add a dummy PDF for testing if it doesn't exist
    dummy_pdf_path = "../data/example.pdf"
    if not os.path.exists("../data"):
        os.makedirs("../data")
    if not os.path.exists(dummy_pdf_path):
        print(f"Creating a dummy PDF at {dummy_pdf_path} for testing utils.py...")
        try:
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(dummy_pdf_path)
            c.drawString(100, 750, "This is a dummy PDF for testing.")
            c.drawString(100, 730, "It contains some sample text to be chunked.")
            for i in range(10):
                c.drawString(100, 710 - i*15, f"Line {i+1}: This is some more content for chunking purposes.")
            c.save()
            print("Dummy PDF created.")
        except ImportError:
            print("ReportLab not installed. Please install it (`pip install reportlab`) or create `data/example.pdf` manually to test chunking.")
            print("Skipping PDF chunking test due to missing dependency.")
            exit()
        except Exception as e:
            print(f"Could not create dummy PDF: {e}. Please ensure `data/example.pdf` exists.")
            exit()

    try:
        chunks = load_and_chunk_pdf(dummy_pdf_path)
        print(f"Successfully loaded and chunked PDF. Number of chunks: {len(chunks)}")
        if chunks:
            print("\nFirst chunk content preview:")
            print(chunks[0].page_content[:500])
            print(f"Source page: {chunks[0].metadata.get('page', 'N/A')}")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure 'data/example.pdf' exists.")
    except Exception as e:
        print(f"An unexpected error occurred during PDF chunking: {e}")