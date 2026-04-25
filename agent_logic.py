"""
agent_logic.py

Multi-stage LangChain chain for Catalyst AI-Powered Skill Assessment & Learning Agent.

Stages:
  1. Extraction  – pull required skills from the Job Description
  2. Verification – match those skills against the candidate's resume
  3. Recommendation – generate interview questions and a personalised learning plan
"""

from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_llm(model: str = "gpt-4o-mini", temperature: float = 0.3) -> ChatOpenAI:
    """Return a configured ChatOpenAI instance."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it to a .env file or export it in your shell."
        )
    return ChatOpenAI(model=model, temperature=temperature, openai_api_key=api_key)


def _parse_json_block(text: str) -> dict | list:
    """Extract and parse a JSON block from an LLM response string."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Stage 1 – Skill Extraction
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert technical recruiter. "
            "Extract ALL technical and soft skills required by the job description. "
            "Return ONLY a valid JSON object with a single key 'required_skills' "
            "whose value is a list of skill name strings. "
            "No commentary, no markdown fences – pure JSON.",
        ),
        ("human", "Job Description:\n{job_description}"),
    ]
)


def extract_required_skills(job_description: str, llm: ChatOpenAI | None = None) -> list[str]:
    """Stage 1 – Extract required skills from the Job Description."""
    llm = llm or _build_llm()
    chain = _EXTRACTION_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"job_description": job_description})
    data = _parse_json_block(raw)
    return data.get("required_skills", [])


# ---------------------------------------------------------------------------
# Stage 2 – Gap Verification
# ---------------------------------------------------------------------------

_VERIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert technical recruiter performing a skill gap analysis. "
            "Given a list of required skills and a candidate's resume, classify each skill as "
            "'matched' (clearly demonstrated in the resume) or 'missing' (absent or insufficient). "
            "Return ONLY a valid JSON object with two keys: "
            "'matched_skills' (list of strings) and 'missing_skills' (list of strings). "
            "No commentary, no markdown fences – pure JSON.",
        ),
        (
            "human",
            "Required Skills:\n{required_skills}\n\nResume:\n{resume}",
        ),
    ]
)


def identify_skill_gaps(
    required_skills: list[str],
    resume: str,
    llm: ChatOpenAI | None = None,
) -> dict:
    """Stage 2 – Verify which skills are present / missing in the resume.

    Returns a dict with keys 'matched_skills' and 'missing_skills'.
    """
    llm = llm or _build_llm()
    chain = _VERIFICATION_PROMPT | llm | StrOutputParser()
    raw = chain.invoke(
        {
            "required_skills": json.dumps(required_skills),
            "resume": resume,
        }
    )
    data = _parse_json_block(raw)
    return {
        "matched_skills": data.get("matched_skills", []),
        "missing_skills": data.get("missing_skills", []),
    }


# ---------------------------------------------------------------------------
# Stage 3a – Interview Question Generation
# ---------------------------------------------------------------------------

_INTERVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior technical interviewer. "
            "Generate exactly 3 practical, scenario-based interview questions that assess "
            "a candidate's real-world proficiency in the given missing skills. "
            "Each question should be challenging but fair and tied to a specific skill. "
            "Return ONLY a valid JSON object with a single key 'questions' whose value is "
            "a list of objects, each with 'skill' and 'question' fields. "
            "No commentary, no markdown fences – pure JSON.",
        ),
        (
            "human",
            "Missing skills that need assessment:\n{missing_skills}",
        ),
    ]
)


def generate_interview_questions(
    missing_skills: list[str],
    llm: ChatOpenAI | None = None,
) -> list[dict]:
    """Stage 3a – Generate 3 technical interview questions for missing skills.

    Returns a list of dicts with 'skill' and 'question' keys.
    """
    llm = llm or _build_llm()
    chain = _INTERVIEW_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"missing_skills": json.dumps(missing_skills)})
    data = _parse_json_block(raw)
    return data.get("questions", [])


# ---------------------------------------------------------------------------
# Stage 3b – Learning Plan Generation
# ---------------------------------------------------------------------------

_LEARNING_PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert learning & development coach. "
            "Create a personalised learning plan for a candidate who is missing the listed skills. "
            "For each missing skill provide: "
            "(1) 'skill' – the skill name, "
            "(2) 'adjacent_skills' – a list of 2-3 complementary skills that reinforce learning, "
            "(3) 'estimated_weeks' – a realistic integer number of weeks to reach working proficiency, "
            "(4) 'resources' – a list of 2-3 learning resource objects, each with 'title' and 'url' fields "
            "(use placeholder URLs like 'https://example.com/resource' if no real URL is known). "
            "Return ONLY a valid JSON object with a single key 'learning_plan' whose value is a list "
            "of the objects described above. "
            "No commentary, no markdown fences – pure JSON.",
        ),
        (
            "human",
            "Missing skills:\n{missing_skills}\n\nCandidate resume summary:\n{resume_snippet}",
        ),
    ]
)


def generate_learning_plan(
    missing_skills: list[str],
    resume: str,
    llm: ChatOpenAI | None = None,
) -> list[dict]:
    """Stage 3b – Generate a personalised learning path with adjacent skills and resources.

    Returns a list of dicts with 'skill', 'adjacent_skills', 'estimated_weeks', and 'resources'.
    """
    llm = llm or _build_llm()
    chain = _LEARNING_PLAN_PROMPT | llm | StrOutputParser()
    # Truncate to first 1500 chars to stay within context budget while keeping personalisation signal
    raw = chain.invoke(
        {
            "missing_skills": json.dumps(missing_skills),
            "resume_snippet": resume[:1500],
        }
    )
    data = _parse_json_block(raw)
    return data.get("learning_plan", [])


# ---------------------------------------------------------------------------
# Convenience – run the full pipeline
# ---------------------------------------------------------------------------

def run_full_analysis(job_description: str, resume: str) -> dict:
    """Run all three stages and return a consolidated result dict.

    Keys in the returned dict:
      - required_skills  : list[str]
      - matched_skills   : list[str]
      - missing_skills   : list[str]
      - match_score      : float  (0–100, percentage of required skills matched)
      - interview_questions : list[dict]
      - learning_plan    : list[dict]
    """
    llm = _build_llm()

    # Stage 1
    required_skills = extract_required_skills(job_description, llm=llm)

    # Stage 2
    gap_data = identify_skill_gaps(required_skills, resume, llm=llm)
    matched = gap_data["matched_skills"]
    missing = gap_data["missing_skills"]

    # Calculate a simple match score
    total = len(required_skills) if required_skills else 1
    match_score = round(len(matched) / total * 100, 1)

    # Stage 3 (only if there are missing skills)
    if missing:
        questions = generate_interview_questions(missing, llm=llm)
        plan = generate_learning_plan(missing, resume, llm=llm)
    else:
        questions = []
        plan = []

    return {
        "required_skills": required_skills,
        "matched_skills": matched,
        "missing_skills": missing,
        "match_score": match_score,
        "interview_questions": questions,
        "learning_plan": plan,
    }
