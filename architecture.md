# 🏗️ Catalyst – Architecture Documentation

## Overview

Catalyst uses a **multi-stage LLM chain** built on LangChain and OpenAI's GPT-4o-mini model.
Each stage has a dedicated, narrowly scoped prompt so that the LLM produces structured JSON output
that can be parsed deterministically. This avoids the reliability issues that arise from asking a
single monolithic prompt to do everything at once.

---

## Pipeline Diagram

```
┌─────────────────────────────────────────────────┐
│  User Inputs                                    │
│  ┌─────────────────┐   ┌──────────────────────┐ │
│  │ Job Description │   │   Resume (text/PDF)  │ │
│  └────────┬────────┘   └──────────┬───────────┘ │
└───────────┼──────────────────────┼─────────────┘
            │                      │
            ▼                      │
   ┌────────────────────┐          │
   │   STAGE 1          │          │
   │   Extraction       │          │
   │                    │          │
   │  Prompt: "Extract  │          │
   │  all required      │          │
   │  skills from JD"   │          │
   │                    │          │
   │  Output:           │          │
   │  required_skills[] │          │
   └────────┬───────────┘          │
            │                      │
            └──────────┬───────────┘
                       ▼
          ┌────────────────────────┐
          │   STAGE 2              │
          │   Baseline Verification│
          │                        │
          │  Prompt: "Compare      │
          │  required skills to    │
          │  resume; classify each │
          │  as matched/missing"   │
          │                        │
          │  Output:               │
          │  baseline matched[]    │
          │  baseline missing[]    │
          └──────────┬─────────────┘
                     │
                     ▼
          ┌────────────────────────┐
          │   STAGE 3              │
          │   Conversational Qs    │
          │                        │
          │  Prompt: "Ask practical│
          │  questions per required│
          │  skill"                │
          │                        │
          │  Output:               │
          │  assessment_questions[]│
          └──────────┬─────────────┘
                     │ candidate answers
                     ▼
          ┌────────────────────────┐
          │   STAGE 4              │
          │   Proficiency Scoring  │
          │                        │
          │  Prompt: "Use resume + │
          │  answers to rate each  │
          │  required skill"       │
          │                        │
          │  Output:               │
          │  skill_assessment[]    │
          │  matched_skills[]      │
          │  missing_skills[]      │
          └──────────┬─────────────┘
                     │
           ┌─────────┴──────────┐
           ▼                    ▼
 ┌──────────────────┐  ┌──────────────────────┐
 │  STAGE 5a        │  │  STAGE 5b            │
 │  Interview       │  │  Learning Plan       │
 │  Questions       │  │  Generator           │
 │                  │  │                      │
 │  Prompt: "Write  │  │  Prompt: "Create a   │
 │  3 scenario-     │  │  realistic adjacent- │
 │  based questions │  │  skill learning plan │
 │  for remaining   │  │  with time + links"  │
 │  gaps"           │  │                      │
 │                  │  │                      │
 │  Output:         │  │  Output:             │
 │  questions[]     │  │  learning_plan[]     │
 └──────────────────┘  └──────────────────────┘
```

---

## Stage Details

### Stage 1 – Extraction

**Goal:** Identify every technical and soft skill mentioned in the Job Description.

**Input:** Raw Job Description text

**Prompt strategy:** The system prompt instructs the model to act as a technical recruiter and
return a pure JSON object. Constraining the output format ensures downstream stages can parse the
result reliably without regex heuristics.

**Output schema:**
```json
{
  "required_skills": ["Python", "FastAPI", "Docker", "Kubernetes", "PostgreSQL"]
}
```

---

### Stage 2 – Baseline Verification

**Goal:** Build an initial resume-only view of likely matched vs missing skills.

**Inputs:**
- `required_skills[]` (from Stage 1)
- Resume text

**Prompt strategy:** The model is instructed to compare each skill against evidence in the resume.
This stage is used as a baseline and to prioritize which skills need deeper conversational probing.

**Output schema:**
```json
{
  "matched_skills": ["Python", "Docker"],
  "missing_skills": ["FastAPI", "Kubernetes", "PostgreSQL"]
}
```

**Derived metric:**
```
match_score = (len(matched_skills) / len(required_skills)) × 100
```

---

### Stage 3 – Conversational Assessment Questions

**Goal:** Ask practical, skill-specific questions to validate claimed proficiency.

**Inputs:**
- `required_skills[]` (from Stage 1)
- `missing_skills[]` baseline hints (from Stage 2)
- Resume text

**Prompt strategy:** Generate concise, scenario-oriented questions that can be answered with real
work examples, trade-offs, and outcomes.

---

### Stage 4 – Proficiency Scoring from Answers

**Goal:** Re-score each required skill using both resume evidence and candidate answers.

**Inputs:**
- `required_skills[]`
- Resume text
- Assessment transcript (question/answer pairs)

**Output fields include:**
- `skill_assessment[]` with per-skill proficiency level, confidence, evidence, and gap reason
- `matched_skills[]` and `missing_skills[]` updated from conversational evidence

---

### Stage 5a – Interview Question Generation

**Goal:** Produce three practical, scenario-based interview questions that probe real proficiency
in the missing skills.

**Input:** `missing_skills[]` (from Stage 2)

**Prompt strategy:** The model is asked for scenario-based questions (rather than factual quizzes)
to surface practical, on-the-job knowledge. Exactly three questions are requested so the output is
consistent and concise.

**Output schema:**
```json
{
  "questions": [
    { "skill": "FastAPI", "question": "Describe how you would implement OAuth2 …" },
    { "skill": "Kubernetes", "question": "Walk me through how you would debug a CrashLoopBackOff …" },
    { "skill": "PostgreSQL", "question": "How would you design an indexing strategy for …" }
  ]
}
```

---

### Stage 5b – Learning Plan Generation

**Goal:** Build a personalised upskilling roadmap for each missing skill.

**Inputs:**
- `missing_skills[]` (from Stage 4)
- `matched_skills[]` (from Stage 4)
- Resume snippet (first 1 500 characters, for personalisation context)

**Prompt strategy:** The model is asked for adjacent skills (to broaden the candidate's mental
model), an estimated time-to-proficiency, and resource links. Placeholders are used for URLs when
the model cannot confirm an authoritative source.

**Output schema:**
```json
{
  "learning_plan": [
    {
      "skill": "FastAPI",
      "adjacent_skills": ["Pydantic", "Starlette", "Async Python"],
      "estimated_weeks": 3,
      "resources": [
        { "title": "FastAPI Official Tutorial", "url": "https://fastapi.tiangolo.com/tutorial/" },
        { "title": "Full Stack FastAPI Template", "url": "https://github.com/tiangolo/full-stack-fastapi-template" }
      ]
    }
  ]
}
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **Separate prompts per stage** | Smaller, focused prompts produce more accurate and consistent JSON than monolithic "do-everything" prompts. |
| **JSON-only output constraint** | Eliminates the need for regex parsing and reduces hallucination noise in the structured fields. |
| **GPT-4o-mini** | Balances quality and cost for a hackathon prototype. Swap `model` in `_build_llm()` for a stronger model if needed. |
| **Resume truncation at 1 500 chars (Stage 3b)** | Keeps the prompt within context budget while still providing enough personalisation signal. |
| **Single shared `ChatOpenAI` instance** | Reduces object-creation overhead when all stages run sequentially in `run_full_analysis()`. |
| **`python-dotenv` for secrets** | Keeps API keys out of source control. |

---

## Extending the Pipeline

- **Add a Stage 4 (Scoring Rubric):** After interview questions are answered (if collecting live responses), a new stage could score the answers and adjust the `match_score`.
- **Async parallelism:** Stages 3a and 3b are independent and could be executed concurrently using `asyncio.gather` and LangChain's async chain API for lower latency.
- **Retrieval-Augmented Generation (RAG):** Replace placeholder resource URLs with real links by adding a web search tool (e.g., Tavily or SerpAPI) via LangChain's tool-calling interface.
- **Vector memory:** Store past analyses in a vector database (e.g., Chroma or Pinecone) so returning users get progressively personalised recommendations.
