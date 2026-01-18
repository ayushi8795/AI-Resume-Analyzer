REPORT_SYSTEM = """You are an expert resume reviewer and hiring manager.
Be direct, practical, and specific. Use simple English.
Only use evidence from the provided context when referencing the resume."""

REPORT_USER = """Task:
Given a job description and resume context, create a structured evaluation.

Return JSON with keys:
- match_score (0-100 integer)
- strong_matches (list of strings)
- missing_skills (list of strings)
- resume_risks (list of strings)  # unclear points, gaps, weak metrics
- bullet_improvements (list of strings)  # rewrite suggestions
- tailored_summary (string)  # 3-5 lines
- recommended_projects (list of strings) # 2 project ideas aligned to JD

Job Description:
{job_description}

Extracted NLP Signals:
- resume_skills: {resume_skills}
- jd_skills: {jd_skills}
- years_experience_hint: {years_exp}

Resume Context (top relevant chunks):
{context}
"""
