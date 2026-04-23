from datetime import datetime, timezone
from uuid import uuid4

from agents.models.config import ORCHESTRATOR_SEED
from agents.models.models import AgentResponse, MedicalAgentState
from agents.orchestrator.chat_protocol import AGENT_DISPLAY_NAMES, chat_proto
from agents.services.llm_service import ASI_MODEL, get_llm_client
from agents.services.state_service import state_service
from uagents import Agent, Context, Model
from uagents_core.contrib.protocols.chat import ChatMessage, EndSessionContent, TextContent

orchestrator = Agent(
    name="medical-orchestrator",
    seed=ORCHESTRATOR_SEED,
    port=8000,
    mailbox=True,
    publish_agent_details=True,
)

orchestrator.include(chat_proto, publish_manifest=True)

SYNTHESIS_PROMPT = """You are a senior physician synthesizing specialist assessments into a unified clinical summary for a patient.

Your job:
- Integrate all specialist reports into one coherent, non-repetitive response
- Lead with the urgency level from the risk assessment
- Provide a clear, prioritized action plan
- Write in plain language (explain any medical terms used)

Format your response exactly like this:

## 🏥 Medical Assessment Summary

**Urgency:** [CRITICAL / URGENT / SOON / ROUTINE / MONITORING]

## Key Findings
[Bullet points synthesizing the most important findings across all reports]

## Action Plan
[Numbered steps, most urgent first]

## Warning Signs — Seek Immediate Care If:
[Bullet list of red flags]

---
⚠️ *This assessment is AI-generated and for informational purposes only. Always consult a licensed healthcare provider before making any medical decisions.*"""


class HealthResponse(Model):
    status: str


@orchestrator.on_rest_get("/health", HealthResponse)
async def health(ctx: Context) -> HealthResponse:
    return HealthResponse(status="ok")


@orchestrator.on_message(AgentResponse)
async def handle_agent_response(ctx: Context, sender: str, response: AgentResponse):
    ctx.logger.info(f"Response from {response.agent_type} | session={response.chat_session_id}")

    state = state_service.get_state(response.chat_session_id)
    if state is None:
        ctx.logger.error(f"No state for session {response.chat_session_id}")
        return

    if response.agent_type == "symptom":
        state.symptom_analysis = response.result
    elif response.agent_type == "drug":
        state.drug_interaction_analysis = response.result
    elif response.agent_type == "lab":
        state.lab_interpretation = response.result
    elif response.agent_type == "risk":
        state.risk_assessment = response.result
    elif response.agent_type == "insurance":
        state.insurance_guidance = response.result

    if response.agent_type not in state.agents_responded:
        state.agents_responded.append(response.agent_type)

    state_service.set_state(response.chat_session_id, state)
    ctx.logger.info(f"Progress: {len(state.agents_responded)}/{len(state.agents_to_call)} — {state.agents_responded}")

    if set(state.agents_responded) >= set(state.agents_to_call):
        ctx.logger.info("All agents responded. Synthesizing final response...")
        final = await synthesize(state)
        agent_names = " · ".join(AGENT_DISPLAY_NAMES[a] for a in state.agents_to_call)
        full_response = f"**Consulted:** {agent_names}\n\n{final}"
        await ctx.send(
            state.user_sender_address,
            ChatMessage(
                timestamp=datetime.now(tz=timezone.utc),
                msg_id=uuid4(),
                content=[
                    TextContent(type="text", text=full_response),
                    EndSessionContent(type="end-session"),
                ],
            ),
        )


async def synthesize(state: MedicalAgentState) -> str:
    parts = []
    if state.symptom_analysis:
        parts.append(f"**Symptom Analysis:**\n{state.symptom_analysis}")
    if state.drug_interaction_analysis:
        parts.append(f"**Drug Interaction Review:**\n{state.drug_interaction_analysis}")
    if state.lab_interpretation:
        parts.append(f"**Lab Interpretation:**\n{state.lab_interpretation}")
    if state.risk_assessment:
        parts.append(f"**Risk Assessment:**\n{state.risk_assessment}")
    if state.insurance_guidance:
        parts.append(f"**Insurance & Healthcare Navigation:**\n{state.insurance_guidance}")

    combined = "\n\n---\n\n".join(parts)

    client = get_llm_client()
    response = await client.chat.completions.create(
        model=ASI_MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIS_PROMPT},
            {
                "role": "user",
                "content": f"Patient query: {state.query}\n\nSpecialist reports:\n\n{combined}",
            },
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    orchestrator.run()
