from __future__ import annotations
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from .prompts import REPORT_SYSTEM, REPORT_USER

class ResumeReport(BaseModel):
    match_score: int = Field(ge=0, le=100)
    strong_matches: List[str]
    missing_skills: List[str]
    resume_risks: List[str]
    bullet_improvements: List[str]
    tailored_summary: str
    recommended_projects: List[str]

def generate_report(
    model_name: str,
    job_description: str,
    context: str,
    resume_skills: List[str],
    jd_skills: List[str],
    years_exp: int,
) -> Dict[str, Any]:
    llm = ChatOpenAI(model=model_name, temperature=0.2)
    parser = PydanticOutputParser(pydantic_object=ResumeReport)

    prompt = ChatPromptTemplate.from_messages([
        ("system", REPORT_SYSTEM),
        ("user", REPORT_USER + "\n\n{format_instructions}"),
    ])

    msg = prompt.format_messages(
        job_description=job_description,
        context=context,
        resume_skills=resume_skills,
        jd_skills=jd_skills,
        years_exp=years_exp,
        format_instructions=parser.get_format_instructions(),
    )

    out = llm.invoke(msg)
    report = parser.parse(out.content)
    return report.model_dump()
