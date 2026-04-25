import json

import agent_logic
from langchain_core.language_models.fake_chat_models import FakeListChatModel


def test_run_full_analysis_resume_only_flow(monkeypatch):
    """Validates baseline resume-only path and output structure."""
    fake = FakeListChatModel(
        responses=[
            json.dumps({"required_skills": ["Python", "FastAPI", "Kubernetes"]}),
            json.dumps({"matched_skills": ["Python"], "missing_skills": ["FastAPI", "Kubernetes"]}),
            json.dumps(
                {
                    "questions": [
                        {
                            "skill": "FastAPI",
                            "question": "How do you structure dependency injection in FastAPI?",
                            "why_it_matters": "Checks architecture depth.",
                        },
                        {
                            "skill": "Kubernetes",
                            "question": "How do you debug CrashLoopBackOff?",
                            "why_it_matters": "Checks production debugging.",
                        },
                    ]
                }
            ),
            json.dumps(
                {
                    "questions": [
                        {"skill": "FastAPI", "question": "How would you secure auth in FastAPI?"},
                        {"skill": "Kubernetes", "question": "How would you tune CPU/memory requests?"},
                        {"skill": "Kubernetes", "question": "How do you run a safe rolling deployment?"},
                    ]
                }
            ),
            json.dumps(
                {
                    "learning_plan": [
                        {
                            "skill": "FastAPI",
                            "adjacent_skills": ["Pydantic", "AsyncIO"],
                            "estimated_weeks": 3,
                            "resources": [
                                {
                                    "title": "FastAPI docs",
                                    "url": "https://fastapi.tiangolo.com/tutorial/",
                                },
                                {
                                    "title": "Pydantic docs",
                                    "url": "https://docs.pydantic.dev/latest/",
                                },
                            ],
                        },
                        {
                            "skill": "Kubernetes",
                            "adjacent_skills": ["Helm", "Observability"],
                            "estimated_weeks": 5,
                            "resources": [
                                {
                                    "title": "Kubernetes docs",
                                    "url": "https://kubernetes.io/docs/home/",
                                },
                                {"title": "Helm docs", "url": "https://helm.sh/docs/"},
                            ],
                        },
                    ]
                }
            ),
        ]
    )

    monkeypatch.setattr(agent_logic, "_build_llm", lambda: fake)

    result = agent_logic.run_full_analysis(
        job_description="Need Python, FastAPI, and Kubernetes experience.",
        resume="Python engineer with backend API experience.",
    )

    assert result["assessment_mode"] == "resume_only"
    assert result["required_skills"] == ["Python", "FastAPI", "Kubernetes"]
    assert result["matched_skills"] == ["Python"]
    assert result["missing_skills"] == ["FastAPI", "Kubernetes"]
    assert result["match_score"] == 33.3
    assert len(result["assessment_questions"]) == 2
    assert result["skill_assessment"] == []
    assert len(result["interview_questions"]) == 3
    assert len(result["learning_plan"]) == 2


def test_run_full_analysis_conversational_flow(monkeypatch):
    """Validates answer-based rescoring and downstream recommendations."""
    fake = FakeListChatModel(
        responses=[
            json.dumps({"required_skills": ["Python", "FastAPI", "Kubernetes"]}),
            json.dumps({"matched_skills": ["Python"], "missing_skills": ["FastAPI", "Kubernetes"]}),
            json.dumps(
                {
                    "questions": [
                        {
                            "skill": "FastAPI",
                            "question": "How do you structure dependency injection in FastAPI?",
                            "why_it_matters": "Checks architecture depth.",
                        },
                        {
                            "skill": "Kubernetes",
                            "question": "How do you debug CrashLoopBackOff?",
                            "why_it_matters": "Checks production debugging.",
                        },
                    ]
                }
            ),
            json.dumps(
                {
                    "skill_assessment": [
                        {
                            "skill": "Python",
                            "proficiency_level": "strong",
                            "confidence": 88,
                            "evidence": "Solid examples.",
                            "gap_reason": "",
                        },
                        {
                            "skill": "FastAPI",
                            "proficiency_level": "working",
                            "confidence": 74,
                            "evidence": "Good practical answer.",
                            "gap_reason": "Could deepen auth patterns.",
                        },
                        {
                            "skill": "Kubernetes",
                            "proficiency_level": "beginner",
                            "confidence": 62,
                            "evidence": "Basic concepts known.",
                            "gap_reason": "Limited production ownership.",
                        },
                    ],
                    "matched_skills": ["Python", "FastAPI"],
                    "missing_skills": ["Kubernetes"],
                }
            ),
            json.dumps(
                {
                    "questions": [
                        {
                            "skill": "Kubernetes",
                            "question": "How would you roll out canary deployments safely?",
                        },
                        {
                            "skill": "Kubernetes",
                            "question": "How do readiness/liveness probes affect restarts?",
                        },
                        {
                            "skill": "Kubernetes",
                            "question": "How do you tune resource requests and limits?",
                        },
                    ]
                }
            ),
            json.dumps(
                {
                    "learning_plan": [
                        {
                            "skill": "Kubernetes",
                            "adjacent_skills": ["Helm", "SRE fundamentals"],
                            "estimated_weeks": 5,
                            "resources": [
                                {
                                    "title": "Kubernetes basics",
                                    "url": "https://kubernetes.io/docs/tutorials/kubernetes-basics/",
                                },
                                {"title": "Helm docs", "url": "https://helm.sh/docs/"},
                            ],
                        }
                    ]
                }
            ),
        ]
    )

    monkeypatch.setattr(agent_logic, "_build_llm", lambda: fake)

    result = agent_logic.run_full_analysis(
        job_description="Need Python, FastAPI, and Kubernetes experience.",
        resume="Python engineer with backend API experience.",
        assessment_answers=[
            {
                "skill": "FastAPI",
                "question": "How do you structure dependency injection in FastAPI?",
                "answer": "I use Depends, service layers, and test boundaries.",
            },
            {
                "skill": "Kubernetes",
                "question": "How do you debug CrashLoopBackOff?",
                "answer": "I inspect events, logs, probes, and resource pressure before patching.",
            },
        ],
    )

    assert result["assessment_mode"] == "conversational"
    assert result["required_skills"] == ["Python", "FastAPI", "Kubernetes"]
    assert result["matched_skills"] == ["Python", "FastAPI"]
    assert result["missing_skills"] == ["Kubernetes"]
    assert result["match_score"] == 66.7
    assert len(result["assessment_questions"]) == 2
    assert len(result["skill_assessment"]) == 3
    assert len(result["interview_questions"]) == 3
    assert len(result["learning_plan"]) == 1


def test_run_full_analysis_no_missing_skills(monkeypatch):
    """Ensures no-gap path returns empty interview questions and learning plan."""
    fake = FakeListChatModel(
        responses=[
            json.dumps({"required_skills": ["Python", "FastAPI"]}),
            json.dumps({"matched_skills": ["Python", "FastAPI"], "missing_skills": []}),
            json.dumps(
                {
                    "questions": [
                        {
                            "skill": "Python",
                            "question": "How do you design maintainable Python modules?",
                            "why_it_matters": "Checks engineering rigor.",
                        },
                        {
                            "skill": "FastAPI",
                            "question": "How do you structure validation and error handling in FastAPI?",
                            "why_it_matters": "Checks API quality mindset.",
                        },
                    ]
                }
            ),
        ]
    )

    monkeypatch.setattr(agent_logic, "_build_llm", lambda: fake)

    result = agent_logic.run_full_analysis(
        job_description="Need Python and FastAPI.",
        resume="Built production APIs in Python and FastAPI for years.",
    )

    assert result["assessment_mode"] == "resume_only"
    assert result["required_skills"] == ["Python", "FastAPI"]
    assert result["matched_skills"] == ["Python", "FastAPI"]
    assert result["missing_skills"] == []
    assert result["match_score"] == 100.0
    assert len(result["assessment_questions"]) == 2
    assert result["interview_questions"] == []
    assert result["learning_plan"] == []
