# rag-pdf-chatbot
RAG (Retrieval-Augmented Generation) PDF Chatbot - Langchain, LLM, FAISS/Chroma, Streamlit, Render/GCP, Docker and CI-CD GitHub Actions


# Important

    poetry shell
    poetry init, add, install, update

Remove venv

    rm -rf .venv

#### Very important
Name the ```package``` name in ```.toml``` same as the app folder we work in (containing ```.py``` files) i.e ```app``` for our instance


# Main cmd

    poetry run streamlit run app/main.py


docker build -t rag-pdf-chatbot .


docker run -p 8501:8501 --env-file .env rag-pdf-chatbot
