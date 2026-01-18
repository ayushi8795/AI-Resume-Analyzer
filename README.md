# AI-Resume-Analyzer
A practical GenAI project that analyzes a **resume PDF** against a **job description** using:
- **NLP (spaCy)** for skills/entities extraction
- **RAG (Retrieval-Augmented Generation)** using OpenAI embeddings
- **Chroma vector database** for semantic search
- **OpenAI LLM** for a structured hiring-style report
- **Streamlit UI** for a clean demo

## Create Virtual Environment and install packages
- python -m venv envresume
- envresume\Scripts\activate
- pip install -r requirements.txt
- python -m spacy download en_core_web_sm

## Run application
- streamlit run app.py

## Tech Stack
- Python
- Streamlit (UI)
- spaCy + RapidFuzz (NLP)
- LangChain (pipeline)
- OpenAI (Embeddings + LLM)
- ChromaDB (Vector store)
- PyPDF (PDF reading)

## How the App Works

- 1. User uploads resume PDF

- 2. User pastes job description

- 3. Resume text is extracted and cleaned

- 4. NLP extracts skills, entities, and experience

- 5. Text is chunked for embeddings

- 6. OpenAI embeddings stored in Chroma

- 7. Retriever finds relevant resume parts

- 8. LLM generates structured report

- 9. Streamlit displays results

## Output Report Includes

- Match Score (0–100)

- Strong Matches

- Missing Skills

- Resume Risks

- Bullet Improvements

- Tailored Summary

- Recommended Projects