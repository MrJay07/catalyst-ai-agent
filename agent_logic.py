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
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_llm(model: str = "gemini-2.0-flash", temperature: float = 0.3) -> Any:
    """Return a configured chat model instance for the selected provider."""
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise EnvironmentError(
                "Gemini provider selected but dependency is missing. "
                "Install it with: pip install langchain-google-genai"
            ) from exc

        gemini_model = os.getenv("LLM_MODEL", model)
        gemini_key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise EnvironmentError(
                "Gemini provider selected but API key is missing. "
                "Set GEMINI_API_KEY or LLM_API_KEY in your .env file."
            )

        return ChatGoogleGenerativeAI(
            model=gemini_model,
            temperature=temperature,
            google_api_key=gemini_key,
        )

    openai_model = os.getenv("LLM_MODEL", model)
    openai_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise EnvironmentError(
            "OpenAI API key is not set. "
            "Set OPENAI_API_KEY or LLM_API_KEY in your .env file."
        )

    base_url = os.getenv("LLM_BASE_URL")
    if base_url:
        return ChatOpenAI(
            model=openai_model,
            temperature=temperature,
            openai_api_key=openai_key,
            base_url=base_url,
        )

    return ChatOpenAI(model=openai_model, temperature=temperature, openai_api_key=openai_key)


def _parse_json_block(text: str) -> dict | list:
    """Extract and parse the first JSON object/array from an LLM response string."""
    # Strip markdown code fences if present.
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    # First attempt: the full response is already valid JSON.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: locate and decode the first JSON object/array inside extra prose.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[i:])
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            continue

    snippet = cleaned[:200].replace("\n", " ")
    raise ValueError(f"Model did not return valid JSON. Response starts with: {snippet!r}")


def _parse_json_block_or_default(text: str, default: dict | list) -> dict | list:
    """Parse JSON from model output, returning a safe default on malformed responses."""
    try:
        return _parse_json_block(text)
    except ValueError:
        return default


def _normalize_skill_list(items: object) -> list[str]:
    """Return a de-duplicated, cleaned list of non-empty skill strings."""
    if not isinstance(items, list):
        return []

    seen: set[str] = set()
    normalized: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        skill = item.strip()
        if not skill:
            continue
        key = skill.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(skill)
    return normalized


def _get_list_value(data: object, key: str) -> list[Any]:
    """Safely read a list value from a parsed JSON dict payload."""
    if not isinstance(data, dict):
        return []
    value = data.get(key, [])
    return value if isinstance(value, list) else []


def _ordered_skill_match_sets(
    required_skills: list[str],
    matched_skills: list[str],
    missing_skills: list[str],
) -> tuple[list[str], list[str]]:
    """Keep only required skills and preserve job-description ordering."""
    required_lut = {skill.casefold(): skill for skill in required_skills}
    matched_set = {s.casefold() for s in matched_skills if s.casefold() in required_lut}
    missing_set = {s.casefold() for s in missing_skills if s.casefold() in required_lut}

    matched = [skill for skill in required_skills if skill.casefold() in matched_set]
    missing = [
        skill
        for skill in required_skills
        if skill.casefold() in missing_set or skill.casefold() not in matched_set
    ]
    return matched, missing


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


def extract_required_skills(job_description: str, llm: Any | None = None) -> list[str]:
    """Stage 1 – Extract required skills from the Job Description."""
    llm = llm or _build_llm()
    chain = _EXTRACTION_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"job_description": job_description})
    data = _parse_json_block_or_default(raw, {"required_skills": []})
    if not isinstance(data, dict):
        return []
    return _normalize_skill_list(data.get("required_skills", []))


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
    llm: Any | None = None,
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
    data = _parse_json_block_or_default(raw, {"matched_skills": [], "missing_skills": []})
    if not isinstance(data, dict):
        return {"matched_skills": [], "missing_skills": []}
    return {
        "matched_skills": _normalize_skill_list(data.get("matched_skills", [])),
        "missing_skills": _normalize_skill_list(data.get("missing_skills", [])),
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
    llm: Any | None = None,
) -> list[dict]:
    """Stage 3a – Generate 3 technical interview questions for missing skills.

    Returns a list of dicts with 'skill' and 'question' keys.
    """
    llm = llm or _build_llm()
    chain = _INTERVIEW_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"missing_skills": json.dumps(missing_skills)})
    data = _parse_json_block_or_default(raw, {"questions": []})
    return _get_list_value(data, "questions")


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
            "Missing skills:\n{missing_skills}\n\n"
            "Candidate strengths already demonstrated:\n{matched_skills}\n\n"
            "Candidate resume summary:\n{resume_snippet}",
        ),
    ]
)


def generate_learning_plan(
    missing_skills: list[str],
    resume: str,
    matched_skills: list[str] | None = None,
    llm: Any | None = None,
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
            "matched_skills": json.dumps(matched_skills or []),
            "resume_snippet": resume[:1500],
        }
    )
    data = _parse_json_block_or_default(raw, {"learning_plan": []})
    return _get_list_value(data, "learning_plan")


# ---------------------------------------------------------------------------
# Stage 3c – Conversational Proficiency Assessment
# ---------------------------------------------------------------------------

_ASSESSMENT_QUESTIONS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior hiring manager running a conversational skill assessment. "
            "Generate concise, practical questions to verify real proficiency in required skills. "
            "Ask up to 1 question per skill (maximum 6 total). "
            "Prioritize skills that are missing or weakly evidenced in the resume. "
            "Return ONLY valid JSON with key 'questions' containing objects with keys: "
            "'skill', 'question', and 'why_it_matters'. No markdown fences.",
        ),
        (
            "human",
            "Required skills:\n{required_skills}\n\n"
            "Skills likely missing from resume:\n{missing_skills}\n\n"
            "Resume:\n{resume}",
        ),
    ]
)


def generate_assessment_questions(
    required_skills: list[str],
    resume: str,
    missing_skills: list[str] | None = None,
    llm: Any | None = None,
) -> list[dict[str, str]]:
    """Generate a short conversational interview question set per required skill."""
    llm = llm or _build_llm()
    chain = _ASSESSMENT_QUESTIONS_PROMPT | llm | StrOutputParser()
    raw = chain.invoke(
        {
            "required_skills": json.dumps(required_skills),
            "missing_skills": json.dumps(missing_skills or []),
            "resume": resume[:2000],
        }
    )
    data = _parse_json_block_or_default(raw, {"questions": []})
    questions = _get_list_value(data, "questions")

    cleaned: list[dict[str, str]] = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        skill = str(item.get("skill", "")).strip()
        question = str(item.get("question", "")).strip()
        why = str(item.get("why_it_matters", "")).strip()
        if not skill or not question:
            continue
        cleaned.append({"skill": skill, "question": question, "why_it_matters": why})

    return cleaned[:6]


_PROFICIENCY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are evaluating true skill proficiency using resume evidence and interview answers. "
            "Rate each required skill with one of: strong, working, beginner, missing. "
            "Return ONLY valid JSON with keys: "
            "'skill_assessment' (list of objects with keys 'skill', 'proficiency_level', "
            "'confidence' integer 0-100, 'evidence', 'gap_reason'), "
            "'matched_skills' (strong/working), and 'missing_skills' (beginner/missing). "
            "No markdown fences.",
        ),
        (
            "human",
            "Required skills:\n{required_skills}\n\n"
            "Resume:\n{resume}\n\n"
            "Assessment transcript (question/answer pairs):\n{assessment_transcript}",
        ),
    ]
)


def assess_skill_proficiency(
    required_skills: list[str],
    resume: str,
    assessment_answers: list[dict[str, str]],
    llm: Any | None = None,
) -> dict[str, Any]:
    """Score proficiency per required skill from resume evidence + interview answers."""
    llm = llm or _build_llm()
    chain = _PROFICIENCY_PROMPT | llm | StrOutputParser()
    raw = chain.invoke(
        {
            "required_skills": json.dumps(required_skills),
            "resume": resume[:2500],
            "assessment_transcript": json.dumps(assessment_answers),
        }
    )

    data = _parse_json_block_or_default(
        raw,
        {"skill_assessment": [], "matched_skills": [], "missing_skills": []},
    )
    if not isinstance(data, dict):
        return {"skill_assessment": [], "matched_skills": [], "missing_skills": []}

    skill_assessment = _get_list_value(data, "skill_assessment")
    cleaned_assessment: list[dict[str, Any]] = []
    for row in skill_assessment:
        if not isinstance(row, dict):
            continue
        skill = str(row.get("skill", "")).strip()
        level = str(row.get("proficiency_level", "")).strip().lower()
        if not skill or level not in {"strong", "working", "beginner", "missing"}:
            continue

        confidence_raw = row.get("confidence", 0)
        confidence = 0
        if isinstance(confidence_raw, (int, float)):
            confidence = max(0, min(100, int(confidence_raw)))

        cleaned_assessment.append(
            {
                "skill": skill,
                "proficiency_level": level,
                "confidence": confidence,
                "evidence": str(row.get("evidence", "")).strip(),
                "gap_reason": str(row.get("gap_reason", "")).strip(),
            }
        )

    matched = _normalize_skill_list(data.get("matched_skills", []))
    missing = _normalize_skill_list(data.get("missing_skills", []))
    return {
        "skill_assessment": cleaned_assessment,
        "matched_skills": matched,
        "missing_skills": missing,
    }


# ---------------------------------------------------------------------------
# Convenience – run the full pipeline
# ---------------------------------------------------------------------------

def run_full_analysis(
    job_description: str,
    resume: str,
    assessment_answers: list[dict[str, str]] | None = None,
) -> dict:
    """Run all three stages and return a consolidated result dict.

    Keys in the returned dict:
      - required_skills  : list[str]
      - matched_skills   : list[str]
      - missing_skills   : list[str]
      - match_score      : float  (0–100, percentage of required skills matched)
            - assessment_questions: list[dict]
            - skill_assessment  : list[dict]
            - interview_questions : list[dict]
      - learning_plan    : list[dict]
    """
    llm = _build_llm()

    # Stage 1
    required_skills = extract_required_skills(job_description, llm=llm)

    # Stage 2 (resume-based baseline)
    gap_data = identify_skill_gaps(required_skills, resume, llm=llm)
    baseline_matched, baseline_missing = _ordered_skill_match_sets(
        required_skills,
        gap_data["matched_skills"],
        gap_data["missing_skills"],
    )

    # Stage 3c (conversational): create question set, then optionally rescore from answers.
    assessment_questions = generate_assessment_questions(
        required_skills,
        resume,
        missing_skills=baseline_missing,
        llm=llm,
    )

    skill_assessment: list[dict[str, Any]] = []
    if assessment_answers:
        proficiency = assess_skill_proficiency(required_skills, resume, assessment_answers, llm=llm)
        matched, missing = _ordered_skill_match_sets(
            required_skills,
            proficiency["matched_skills"],
            proficiency["missing_skills"],
        )
        skill_assessment = proficiency["skill_assessment"]
    else:
        matched = baseline_matched
        missing = baseline_missing

    total = len(required_skills) if required_skills else 1
    match_score = round(len(matched) / total * 100, 1)

    # Stage 3 (only if there are missing skills)
    if missing:
        questions = generate_interview_questions(missing, llm=llm)
        plan = generate_learning_plan(missing, resume, matched_skills=matched, llm=llm)
    else:
        questions = []
        plan = []

    return {
        "required_skills": required_skills,
        "matched_skills": matched,
        "missing_skills": missing,
        "match_score": match_score,
        "assessment_questions": assessment_questions,
        "skill_assessment": skill_assessment,
        "assessment_mode": "conversational" if assessment_answers else "resume_only",
        "interview_questions": questions,
        "learning_plan": plan,
    }
