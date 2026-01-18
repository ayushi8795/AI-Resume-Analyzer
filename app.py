import os
import tempfile
import streamlit as st

from src.config import settings
from src.pdf_loader import load_pdf_text
from src.text_cleaner import clean_text
from src.chunking import chunk_text
from src.nlp_extract import extract_skills, extract_entities, estimate_years_experience
from src.vectorstore import build_chroma
from src.llm_report import generate_report

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("AI Resume Analyzer")

st.write("Upload your resume PDF and paste a job description. You will get a match report.")

uploaded = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])
job_description = st.text_area("Paste Job Description", height=220)

run_btn = st.button("Analyze")

if run_btn:
    if not uploaded or not job_description.strip():
        st.error("Please upload a resume PDF and paste a job description.")
        st.stop()

    with st.spinner("Reading PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getvalue())
            pdf_path = tmp.name

        resume_text = load_pdf_text(pdf_path)
        resume_text = clean_text(resume_text)
        jd_text = clean_text(job_description)

    with st.spinner("Running NLP extraction..."):
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)
        years_exp = estimate_years_experience(resume_text)
        ents = extract_entities(resume_text)

    with st.spinner("Building vector store (Chroma) + retrieving relevant context..."):
        resume_chunks = chunk_text(resume_text)
        jd_chunks = chunk_text(jd_text, chunk_size=600, chunk_overlap=100)

        texts = []
        metas = []

        for i, c in enumerate(resume_chunks):
            texts.append(c)
            metas.append({"source": "resume", "chunk_id": i})

        for i, c in enumerate(jd_chunks):
            texts.append(c)
            metas.append({"source": "job_description", "chunk_id": i})

        # Use a unique collection name per run to keep it simple
        collection = f"resume_analyzer_{os.urandom(4).hex()}"

        vs = build_chroma(
            texts=texts,
            metadatas=metas,
            persist_dir=settings.chroma_dir,
            collection_name=collection,
            embed_model=settings.openai_embed_model,
        )

        # Retrieve using the JD as the query
        retriever = vs.as_retriever(search_kwargs={"k": 6})
        docs = retriever.invoke(jd_text)
        context = "\n\n---\n\n".join([d.page_content for d in docs])

    with st.spinner("Generating report with OpenAI..."):
        report = generate_report(
            model_name=settings.openai_model,
            job_description=jd_text,
            context=context,
            resume_skills=resume_skills,
            jd_skills=jd_skills,
            years_exp=years_exp,
        )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Match Score")
        st.metric("Score (0-100)", report["match_score"])

        st.subheader("Strong Matches")
        st.write(report["strong_matches"])

        st.subheader("Missing Skills")
        st.write(report["missing_skills"])

        st.subheader("Resume Risks")
        st.write(report["resume_risks"])

    with col2:
        st.subheader("Tailored Summary")
        st.write(report["tailored_summary"])

        st.subheader("Bullet Improvements")
        st.write(report["bullet_improvements"])

        st.subheader("Recommended Projects")
        st.write(report["recommended_projects"])

    with st.expander("NLP Entities Found (spaCy)", expanded=False):
        st.json(ents)

    with st.expander("Retrieved Context Used by LLM (for transparency)", expanded=False):
        st.write(context)
