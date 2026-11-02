"""Streamlit frontend for the Agentic AI Medication Management System."""
from __future__ import annotations

import json
import os
from typing import Dict, Any, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from main import MedicationManagementSystem
from src.utils.config import SYNTHETIC_PATIENTS_DIR

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Medication Management Dashboard", layout="wide")


def _format_patient_label(patient: Dict[str, Any]) -> str:
    return f"{patient.get('patient_id', 'Unknown')} · {patient.get('name', 'Unnamed')}"


@st.cache_resource(show_spinner=False)
def load_system() -> MedicationManagementSystem:
    """Create a single instance of the multi-agent system for reuse."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("GOOGLE_API_KEY not found. Please set it in your .env file.")
        st.stop()
    return MedicationManagementSystem(google_api_key=google_api_key)


@st.cache_data(show_spinner=False)
def load_patients() -> List[Dict[str, Any]]:
    """Load patient roster from the synthetic dataset."""
    patients_path = SYNTHETIC_PATIENTS_DIR / "patients.json"
    if not patients_path.exists():
        return []
    with open(patients_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def render_patient_overview(report: Dict[str, Any]) -> None:
    patient = report.get("patient", {})
    summary = report.get("schedule_summary", {})

    st.subheader("Patient Overview")
    st.markdown(
        f"**Name:** {patient.get('name', 'Unknown')}  |  **Gender:** {patient.get('gender', 'Not specified')}"
    )

    metrics = st.columns(3)
    metrics[0].metric("Age", patient.get("age", "-"))
    condition_count = len(patient.get("conditions", []))
    metrics[1].metric("Chronic Conditions", condition_count)

    adherence_rate = summary.get("adherence_rate")
    adherence_text = f"{adherence_rate:.1%}" if isinstance(adherence_rate, (int, float)) else "-"
    metrics[2].metric("Adherence", adherence_text)

    if patient.get("conditions"):
        st.write("Conditions: " + ", ".join(patient["conditions"]))
    if patient.get("allergies"):
        st.write("Allergies: " + ", ".join(patient["allergies"]))

    meds = report.get("medications_details", [])
    if meds:
        med_df = pd.DataFrame(meds)
        st.subheader("Active Medications")
        st.dataframe(med_df, use_container_width=True, hide_index=True)


def render_interactions(report: Dict[str, Any]) -> None:
    st.header("Drug Interaction Review")
    interactions = report.get("interactions", [])
    if interactions:
        for interaction in interactions:
            meds = " + ".join(interaction.get("medications", []))
            severity = interaction.get("severity", "unknown").upper()
            description = interaction.get("description", "No description")
            recommendation = interaction.get("recommendation", "No recommendation provided")
            st.warning(
                f"**{meds}**\n\nSeverity: {severity}\n\nDetails: {description}\n\nRecommended Action: {recommendation}"
            )
    else:
        st.success("No potential interactions detected for the current regimen.")


def render_education(report: Dict[str, Any]) -> None:
    st.header("Patient Education")
    query_result = report.get("medication_query", {})
    answer = query_result.get("answer", "No educational content available.")
    st.markdown(answer)

    warnings = query_result.get("warnings", [])
    for warning in warnings:
        st.warning(warning)

    reminder = report.get("reminder")
    if reminder:
        st.info(reminder)


def render_adherence(report: Dict[str, Any]) -> None:
    st.header("Adherence Insights")

    summary = report.get("schedule_summary", {})
    cols = st.columns(3)
    cols[0].metric("Total Doses", summary.get("total_doses", 0))
    cols[1].metric("Doses Taken", summary.get("taken_doses", 0))
    adherence_rate = summary.get("adherence_rate")
    adherence_text = f"{adherence_rate:.1%}" if isinstance(adherence_rate, (int, float)) else "-"
    cols[2].metric("Calculated Adherence", adherence_text)

    missed = report.get("missed_doses", [])
    if missed:
        missed_df = pd.DataFrame(missed)
        st.subheader("Recent Missed Doses")
        st.dataframe(missed_df, use_container_width=True, hide_index=True)

    coaching = report.get("coaching", {})
    if coaching:
        st.subheader("Coaching Highlights")
        st.write(f"Pattern: {coaching.get('pattern', 'Not available')}")
        message = coaching.get("coaching_message")
        if message:
            st.markdown(message)

        strategies = coaching.get("strategies", [])
        if strategies:
            st.write("Suggested Strategies:")
            for strategy in strategies:
                st.write(f"- {strategy}")

    monitoring = report.get("monitoring", {})
    analysis = monitoring.get("adherence_analysis", {})
    if analysis:
        st.subheader("Monitoring Summary")
        trend_cols = st.columns(3)
        overall = analysis.get("overall_adherence")
        recent = analysis.get("recent_adherence")
        total_doses = analysis.get("total_doses", 0)
        missed_doses = analysis.get("missed_doses", 0)
        trend_cols[0].metric("Overall", f"{overall:.1%}" if isinstance(overall, (int, float)) else "-")
        trend_cols[1].metric("Recent", f"{recent:.1%}" if isinstance(recent, (int, float)) else "-")
        trend_cols[2].metric("Missed Doses", missed_doses)

        for warning in analysis.get("warnings", []):
            st.warning(warning.get("message", "Adherence warning"))

    recommendations = monitoring.get("recommendations", [])
    if recommendations:
        st.subheader("Recommended Next Steps")
        for recommendation in recommendations:
            st.write(f"- {recommendation}")


def render_custom_question(system: MedicationManagementSystem, report: Dict[str, Any]) -> None:
    st.header("Ask a Question")
    default_question = "What lifestyle tips support my current medications?"
    user_question = st.text_area("Medication question", default_question)

    if st.button("Ask AI"):
        if not user_question.strip():
            st.warning("Enter a question before submitting.")
            return

        with st.spinner("Generating answer..."):
            try:
                response = system.medication_manager.process_query(
                    user_question.strip(),
                    report.get("patient", {})
                )
                st.session_state["custom_answer"] = response
            except Exception as exc:
                st.session_state["custom_answer_error"] = str(exc)

    if "custom_answer_error" in st.session_state:
        st.error(st.session_state.pop("custom_answer_error"))

    custom_answer = st.session_state.get("custom_answer")
    if custom_answer:
        st.subheader("AI Response")
        st.markdown(custom_answer.get("answer", "No answer available."))
        warnings = custom_answer.get("warnings", [])
        for warning in warnings:
            st.warning(warning)


def main() -> None:
    system = load_system()
    patients = load_patients()

    if not patients:
        st.error("Patient dataset not found. Check the data directory configuration.")
        return

    sidebar_labels = [_format_patient_label(patient) for patient in patients]
    selection = st.sidebar.selectbox("Select patient", sidebar_labels)
    selected_index = sidebar_labels.index(selection)
    selected_patient = patients[selected_index]
    patient_id = selected_patient.get("patient_id")

    st.sidebar.write(f"Patient ID: {patient_id}")

    with st.spinner("Building patient report..."):
        try:
            report = system.generate_patient_report(patient_id)
        except Exception as exc:
            st.error(f"Unable to generate report: {exc}")
            return

    st.title("Medication Management Dashboard")
    st.caption("AI-assisted insights for personalized medication support")

    tab_overview, tab_interactions, tab_education, tab_adherence, tab_questions = st.tabs(
        ["Overview", "Interactions", "Education", "Adherence", "Questions"]
    )

    with tab_overview:
        render_patient_overview(report)
    with tab_interactions:
        render_interactions(report)
    with tab_education:
        render_education(report)
    with tab_adherence:
        render_adherence(report)
    with tab_questions:
        render_custom_question(system, report)


if __name__ == "__main__":
    main()
