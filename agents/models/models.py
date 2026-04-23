from typing import List
from uagents import Model


class MedicalAgentState(Model):
    """
    Shared state flowing from orchestrator to specialist agents.

    The orchestrator populates agents_to_call, dispatches to all of them in parallel,
    and collects AgentResponse messages until all have responded, then synthesizes.
    """

    chat_session_id: str
    query: str
    user_sender_address: str

    agents_to_call: List[str] = []
    agents_responded: List[str] = []
    suspected_conditions: str = ""

    symptom_analysis: str = ""
    drug_interaction_analysis: str = ""
    lab_interpretation: str = ""
    risk_assessment: str = ""
    insurance_guidance: str = ""


class AgentResponse(Model):
    """Sent by each specialist agent back to the orchestrator with its result."""

    chat_session_id: str
    agent_type: str  # "symptom" | "drug" | "lab" | "risk"
    result: str
