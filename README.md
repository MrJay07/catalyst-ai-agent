# ⚡ Catalyst – AI-Powered Skill Assessment & Learning Agent

[![Test Suite](https://github.com/MrJay07/catalyst-ai-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/MrJay07/catalyst-ai-agent/actions/workflows/tests.yml)

Catalyst is a hackathon prototype that uses a **multi-stage LLM chain** to assess real proficiency, not just resume claims. Given a Job Description and a Resume, Catalyst:

1. **Extracts** all required technical and soft skills from the JD.
2. **Builds conversational assessment questions** tailored to likely weak/missing skills.
3. **Evaluates candidate answers** to score proficiency per required skill.
4. **Generates** three targeted interview questions for remaining gaps.
5. **Produces** a personalised learning plan with realistic adjacent skills, time estimates, and curated resource links.

---

## 🖥️ Demo Screenshot

> Run the app locally (see setup below) and open `http://localhost:8501`.

---

## 🏗️ Architecture

See [architecture.md](architecture.md) for a detailed explanation of the multi-stage LLM chain.

```
Job Description + Resume
        │
        ▼
  ┌─────────────┐
  │  Stage 1    │  Skill Extraction  (GPT-4o-mini)
  │  Extraction │  → required_skills[]
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Stage 2    │  Gap Verification  (GPT-4o-mini)
  │ Verification│  → matched_skills[], missing_skills[], match_score
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Stage 3    │  Conversational Assessment  (GPT-4o-mini)
  │  Questions  │  → assessment_questions[]
  └──────┬──────┘
         │ candidate answers
         ▼
  ┌─────────────┐
  │  Stage 4    │  Proficiency Scoring  (GPT-4o-mini)
  │  Evaluate   │  → skill_assessment[], updated gaps
  └──────┬──────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌──────────────┐
│ 5a    │  │ 5b           │
│ Inter-│  │ Learning Plan│
│ view  │  │ Generator    │
│ Qs    │  └──────────────┘
└───────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 1. Clone the repository

```bash
git clone https://github.com/MrJay07/catalyst-ai-agent.git
cd catalyst-ai-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your environment

Create a `.env` file in the project root:

```env
# Provider selection: openai (default) or gemini
LLM_PROVIDER=openai

# Model name for the selected provider
LLM_MODEL=gpt-4o-mini

# Generic key variable (recommended)
LLM_API_KEY=your-key-here

# Optional provider-specific keys (either is fine)
OPENAI_API_KEY=sk-...your-key-here...
GEMINI_API_KEY=your-gemini-key-here

# Optional: OpenAI-compatible base URL (OpenRouter, Groq OpenAI-compatible APIs, etc.)
# LLM_BASE_URL=https://openrouter.ai/api/v1
```

#### Use Gemini API

To use Gemini, set the following in `.env`:

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-1.5-flash
GEMINI_API_KEY=your-gemini-api-key
```

### 5. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
catalyst-ai-agent/
├── app.py            # Streamlit UI entry point
├── agent_logic.py    # Multi-stage LangChain/OpenAI pipeline
├── requirements.txt  # Python dependencies
├── architecture.md   # Architecture documentation
└── README.md         # This file
```

---

## 🧠 Scoring Logic

| Score Range | Label              | Meaning                                              |
|-------------|--------------------|------------------------------------------------------|
| 75–100 %    | Strong Match 🟢    | Candidate demonstrates most required skills          |
| 50–74 %     | Moderate Match 🟡  | Candidate has foundational skills; targeted gaps     |
| 0–49 %      | Needs Development 🔴 | Significant upskilling required before role fit    |

**Formula:**  `match_score = (matched_skills / required_skills) × 100`

---

## 🔧 Tech Stack

| Layer              | Technology                          |
|--------------------|-------------------------------------|
| Web UI             | [Streamlit](https://streamlit.io)   |
| Agent Orchestration| [LangChain](https://langchain.com)  |
| LLM Provider       | OpenAI GPT-4o-mini                  |
| PDF Parsing        | [pypdf](https://pypdf.readthedocs.io) |
| Config Management  | python-dotenv                       |

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss what you'd like to change.

---

## 📜 License

MIT
