"""
app.py

Catalyst AI-Powered Skill Assessment & Learning Agent
Streamlit entry point
"""

from __future__ import annotations

import io
import json

import streamlit as st

# ── Page config must be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title="Catalyst – AI Skill Assessment",
    page_icon="⚡",
    layout="wide",
)

from agent_logic import run_full_analysis  # noqa: E402  (after set_page_config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(uploaded_file) -> str:
    """Return plain text extracted from an uploaded PDF file."""
    try:
        from pypdf import PdfReader
    except ImportError:
        st.error("pypdf is required to parse PDFs. Run: pip install pypdf")
        return ""

    try:
        # getvalue() avoids empty reads on reruns caused by file pointer state.
        pdf_bytes = uploaded_file.getvalue()
        if not pdf_bytes:
            return ""
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def render_score_gauge(score: float) -> None:
    """Display the match score with colour-coded feedback."""
    if score >= 75:
        colour = "green"
        label = "Strong Match 🟢"
    elif score >= 50:
        colour = "orange"
        label = "Moderate Match 🟡"
    else:
        colour = "red"
        label = "Needs Development 🔴"

    st.markdown(
        f"""
        <div style='text-align:center; padding: 12px 0;'>
            <span style='font-size:3rem; font-weight:700; color:{colour};'>{score}%</span><br/>
            <span style='font-size:1.1rem;'>{label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_analysis_error(exc: Exception) -> str:
    """Return a user-friendly message for common provider/API errors."""
    text = str(exc)
    lower = text.lower()

    if "insufficient_quota" in lower or "exceeded your current quota" in lower:
        return (
            "API quota has been exceeded for this key. "
            "Check your provider billing/usage, then retry."
        )

    if "rate limit" in lower or "error code: 429" in lower:
        return "Provider rate limit reached. Please wait a moment and try again."

    if "invalid_api_key" in lower or "incorrect api key" in lower:
        return "Your API key appears invalid. Update GEMINI_API_KEY (or LLM_API_KEY) and retry."

    if "authentication" in lower or "401" in lower:
        return "Provider authentication failed. Verify your API key and account access."

    if "model did not return valid json" in lower:
        return (
            "The model returned an unexpected response format. "
            "Please run the analysis again; if this persists, try a different model in your .env "
            "(for example, LLM_MODEL=gemini-2.0-flash)."
        )

    return f"An unexpected error occurred: {text}"


def run_analysis_or_stop(
    job_description: str,
    resume_text: str,
    assessment_answers: list[dict[str, str]] | None = None,
) -> dict:
    """Run analysis and stop the Streamlit flow with a friendly error when it fails."""
    try:
        return run_full_analysis(
            job_description,
            resume_text,
            assessment_answers=assessment_answers,
        )
    except EnvironmentError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(format_analysis_error(exc))
        st.stop()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("⚡ Catalyst – AI-Powered Skill Assessment & Learning Agent")
    st.markdown(
        "Paste a **Job Description** and your **Resume** below. "
        "Catalyst will extract required skills, run a conversational proficiency check, "
        "identify gaps, and build a realistic personalised learning plan."
    )

    st.divider()

    # ── Input columns ───────────────────────────────────────────────────────
    col_jd, col_resume = st.columns(2)

    with col_jd:
        st.subheader("📋 Job Description")
        job_description = st.text_area(
            "Paste the full job description here:",
            height=320,
            placeholder="e.g. We are looking for a senior Python developer with experience in FastAPI, "
            "Docker, Kubernetes, PostgreSQL, and CI/CD pipelines…",
            key="jd_input",
        )

    with col_resume:
        st.subheader("📄 Resume")
        resume_input_mode = st.radio(
            "Input method:", ["Text", "PDF Upload"], horizontal=True, key="resume_mode"
        )

        resume_text = ""
        if resume_input_mode == "Text":
            resume_text = st.text_area(
                "Paste your resume text here:",
                height=280,
                placeholder="e.g. Experienced Python developer skilled in Flask, Django, "
                "PostgreSQL, Docker, and AWS…",
                key="resume_text_input",
            )
        else:
            uploaded = st.file_uploader("Upload your PDF resume:", type=["pdf"], key="resume_pdf")
            if uploaded:
                resume_text = extract_text_from_pdf(uploaded)
                if resume_text:
                    st.success(f"✅ Extracted {len(resume_text):,} characters from PDF.")
                else:
                    st.warning("Could not extract text from the PDF. Try pasting the text directly.")

    st.divider()

    # ── Step 1: Generate conversational assessment questions ────────────────
    generate_btn = st.button(
        "Generate Assessment Questions",
        type="primary",
        use_container_width=True,
    )

    if generate_btn:
        if not job_description.strip():
            st.error("Please provide a Job Description.")
            st.stop()
        if not resume_text.strip():
            st.error("Please provide your Resume (text or PDF).")
            st.stop()

        with st.spinner("Creating conversational assessment questions…"):
            results = run_analysis_or_stop(job_description, resume_text)

        # Store baseline results and user inputs for step 2.
        st.session_state["job_description"] = job_description
        st.session_state["resume_text"] = resume_text
        st.session_state["results"] = results

    # ── Step 2: candidate answers + answer-aware reassessment ───────────────
    current_results = st.session_state.get("results", {})
    assessment_questions = current_results.get("assessment_questions", [])

    if assessment_questions:
        st.subheader("💬 Conversational Skill Assessment")
        st.caption("Answer each question briefly with concrete examples from your experience.")

        for i, item in enumerate(assessment_questions, start=1):
            skill = item.get("skill", "Skill")
            question = item.get("question", "")
            why = item.get("why_it_matters", "")
            st.markdown(f"**Q{i} ({skill})**: {question}")
            if why:
                st.caption(f"Why this matters: {why}")
            st.text_area(
                f"Your answer for Q{i}",
                height=110,
                key=f"assessment_answer_{i}",
                placeholder="Describe what you did, your approach, trade-offs, and outcome.",
            )

        evaluate_btn = st.button("Evaluate Proficiency & Build Learning Plan")
        if evaluate_btn:
            answered_items: list[dict[str, str]] = []
            for i, item in enumerate(assessment_questions, start=1):
                answer = st.session_state.get(f"assessment_answer_{i}", "").strip()
                if not answer:
                    continue
                answered_items.append(
                    {
                        "skill": item.get("skill", ""),
                        "question": item.get("question", ""),
                        "answer": answer,
                    }
                )

            if not answered_items:
                st.error("Please answer at least one assessment question before evaluation.")
                st.stop()

            with st.spinner("Re-assessing proficiency from your answers…"):
                results = run_analysis_or_stop(
                    st.session_state.get("job_description", job_description),
                    st.session_state.get("resume_text", resume_text),
                    assessment_answers=answered_items,
                )

            st.session_state["results"] = results

    # ── Display results ───────────────────────────────────────────────────────
    if "results" in st.session_state:
        results = st.session_state["results"]
        _render_results(results)


def _render_results(results: dict) -> None:
    """Render the full analysis results."""
    st.success("✅ Analysis complete!")
    st.divider()

    # ── Conversational assessment outcomes ───────────────────────────────────
    st.subheader("🧪 Proficiency Assessment")
    mode = results.get("assessment_mode", "resume_only")
    if mode == "conversational":
        st.caption("Assessment mode: resume + candidate answers")
    else:
        st.caption("Assessment mode: resume-only baseline (answer questions above for deeper validation)")

    skill_assessment = results.get("skill_assessment", [])
    if skill_assessment:
        for row in skill_assessment:
            skill = row.get("skill", "Skill")
            level = row.get("proficiency_level", "unknown").title()
            confidence = row.get("confidence", 0)
            evidence = row.get("evidence", "")
            gap_reason = row.get("gap_reason", "")

            with st.expander(f"{skill} — {level} ({confidence}% confidence)"):
                if evidence:
                    st.markdown(f"**Evidence:** {evidence}")
                if gap_reason:
                    st.markdown(f"**Gap reason:** {gap_reason}")
    else:
        st.info("No per-skill proficiency details yet. Submit assessment answers to generate this view.")

    st.divider()

    # ── Overview metrics ─────────────────────────────────────────────────────
    st.subheader("📊 Skill Match Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Required Skills", len(results["required_skills"]))
    c2.metric("Matched Skills", len(results["matched_skills"]))
    c3.metric("Missing Skills", len(results["missing_skills"]))
    with c4:
        render_score_gauge(results["match_score"])

    st.divider()

    # ── Skills breakdown ─────────────────────────────────────────────────────
    col_matched, col_missing = st.columns(2)

    with col_matched:
        st.subheader("✅ Matched Skills")
        if results["matched_skills"]:
            for skill in results["matched_skills"]:
                st.markdown(f"- {skill}")
        else:
            st.info("No skills matched.")

    with col_missing:
        st.subheader("❌ Missing Skills")
        if results["missing_skills"]:
            for skill in results["missing_skills"]:
                st.markdown(f"- {skill}")
        else:
            st.success("All required skills are present in your resume!")

    st.divider()

    # ── Interview questions ───────────────────────────────────────────────────
    st.subheader("🎤 Targeted Interview Questions")
    if results["interview_questions"]:
        for i, item in enumerate(results["interview_questions"], start=1):
            with st.expander(f"Q{i} – {item.get('skill', 'Skill')}"):
                st.markdown(item.get("question", ""))
    else:
        st.info("No interview questions generated (no skill gaps found).")

    st.divider()

    # ── Learning plan ─────────────────────────────────────────────────────────
    st.subheader("📚 Personalised Learning Plan")
    if results["learning_plan"]:
        for entry in results["learning_plan"]:
            skill_name = entry.get("skill", "Unknown Skill")
            weeks = entry.get("estimated_weeks", "?")
            adjacent = entry.get("adjacent_skills", [])
            resources = entry.get("resources", [])

            with st.expander(f"🎯 {skill_name}  –  ~{weeks} week(s) to proficiency"):
                if adjacent:
                    st.markdown("**Adjacent / Complementary Skills:**")
                    st.markdown(", ".join(f"`{s}`" for s in adjacent))

                if resources:
                    st.markdown("**Curated Resources:**")
                    for res in resources:
                        title = res.get("title", "Resource")
                        url = res.get("url", "#")
                        st.markdown(f"- [{title}]({url})")
    else:
        st.info("No learning plan generated (no skill gaps found).")

    st.divider()

    # ── Raw JSON expander (for debugging / transparency) ─────────────────────
    with st.expander("🔧 Raw JSON Response"):
        st.code(json.dumps(results, indent=2), language="json")


if __name__ == "__main__":
    main()
