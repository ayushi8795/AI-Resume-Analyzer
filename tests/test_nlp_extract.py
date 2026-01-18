from src.nlp_extract import extract_skills, estimate_years_experience

def test_extract_skills_basic():
    text = "I built RAG apps using LangChain, OpenAI, and Chroma on AWS."
    skills = extract_skills(text)
    assert "langchain" in [s.lower() for s in skills]
    assert "openai" in [s.lower() for s in skills]

def test_years_experience():
    text = "I have 5 years of experience in backend development."
    assert estimate_years_experience(text) == 5
