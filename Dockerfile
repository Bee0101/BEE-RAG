FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY rag4.py .
COPY .env .env

RUN pip install --no-cache-dir \
    streamlit \
    google-generativeai \
    langchain \
    langchain-community \
    langchain-text-splitters \
    faiss-cpu \
    python-docx \
    PyPDF2 \
    python-dotenv \
    ollama

EXPOSE 8502

CMD ["streamlit", "run", "rag4.py", "--server.address=0.0.0.0"]


