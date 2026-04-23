import json
from datetime import datetime, timezone
from uuid import uuid4

from uagents import Context, Protocol
from agents.models.config import (
    SYMPTOM_ANALYST_ADDRESS,
    DRUG_INTERACTION_ADDRESS,
    LAB_INTERPRETER_ADDRESS,
    RISK_ASSESSOR_ADDRESS,
    INSURANCE_NAVIGATOR_ADDRESS,
)
from agents.models.models import MedicalAgentState
from agents.services.state_service import state_service
from agents.services.llm_service import get_llm_client, ASI_MODEL
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    TextContent,
    chat_protocol_spec,
)

chat_proto = Protocol(spec=chat_protocol_spec)

AGENT_ADDRESS_MAP = {
    "symptom": SYMPTOM_ANALYST_ADDRESS,
    "drug": DRUG_INTERACTION_ADDRESS,
    "lab": LAB_INTERPRETER_ADDRESS,
    "risk": RISK_ASSESSOR_ADDRESS,
    "insurance": INSURANCE_NAVIGATOR_ADDRESS,
}

AGENT_DISPLAY_NAMES = {
    "symptom": "🔬 Nova (Symptom Analyst)",
    "drug": "💊 Sage (Drug Interaction Checker)",
    "lab": "🧪 Lumen (Lab Interpreter)",
    "risk": "⚠️ Sentinel (Risk Assessor)",
    "insurance": "🏦 Harbor (Insurance Navigator)",
}

ROUTING_PROMPT = """You are a medical triage coordinator deciding which specialist agents to consult.

Available agents:
- "symptom": analyze symptoms and produce a differential diagnosis
- "drug": check drug interactions and medication safety
- "lab": interpret lab results, blood work, or diagnostic test values
- "risk": assess urgency and recommend next steps (ALWAYS include for medical queries)
- "insurance": handle insurance coverage, billing, claims, prior auth, costs, EOBs

Rules:
- If the query is a medical question (symptoms, medications, lab results, health concerns) → include relevant clinical agents, always including "risk"
- If the query mentions insurance, billing, cost, coverage, copay, deductible, prior authorization, claims, or healthcare navigation → include "insurance"
- A query can trigger both clinical agents AND "insurance" at the same time
- If the query is NOT medical or health-related at all (greetings, agent identity, small talk) → return []

Respond with ONLY a valid JSON array. Examples:
- Symptoms only: ["symptom", "risk"]
- Insurance only: ["insurance"]
- Both: ["symptom", "drug", "risk", "insurance"]
- Non-medical: []"""

IDENTITY_RESPONSE = """I'm **MedAgent** — an AI-powered medical multi-agent system built on the Fetch.ai network.

When you describe a concern, I automatically route your query in parallel to a team of specialist AI agents:

- 🔬 **Nova** (Symptom Analyst) — differential diagnosis based on your symptoms
- 💊 **Sage** (Drug Interaction Checker) — flags medication conflicts and contraindications
- 🧪 **Lumen** (Lab Interpreter) — explains blood work and test results in plain English
- ⚠️ **Sentinel** (Risk Assessor) — determines urgency and gives you clear next steps
- 🏦 **Harbor** (Insurance Navigator) — guides you through coverage, billing, and claims

All relevant agents run simultaneously and their findings are synthesized into a single report.

**Try asking something like:**
- *"I have chest pain and I'm on warfarin"*
- *"My hemoglobin is 9.2 and I'm feeling dizzy"*
- *"My insurance denied my claim, what do I do?"*
- *"I have a severe headache and I'm worried about the cost of an MRI"*

⚠️ *This system provides AI-generated information only — always consult a licensed healthcare provider.*"""


async def decide_agents(query: str) -> list:
    client = get_llm_client()
    try:
        response = await client.chat.completions.create(
            model=ASI_MODEL,
            messages=[
                {"role": "system", "content": ROUTING_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        agents = json.loads(content.strip())
        if agents and "risk" not in agents:
            agents.append("risk")
        return [a for a in agents if a in AGENT_ADDRESS_MAP]
    except Exception:
        return ["symptom", "risk"]


@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.now(), acknowledged_msg_id=msg.msg_id),
    )

    text = " ".join(item.text for item in msg.content if isinstance(item, TextContent))
    ctx.logger.info(f"Received query: {text!r}")

    chat_session_id = str(ctx.session)

    agents_to_call = await decide_agents(text)
    ctx.logger.info(f"Routing to agents: {agents_to_call}")

    state = MedicalAgentState(
        chat_session_id=chat_session_id,
        query=text,
        user_sender_address=sender,
        agents_to_call=agents_to_call,
    )
    if not agents_to_call:
        await ctx.send(
            sender,
            ChatMessage(
                timestamp=datetime.now(tz=timezone.utc),
                msg_id=uuid4(),
                content=[
                    TextContent(type="text", text=IDENTITY_RESPONSE),
                    EndSessionContent(type="end-session"),
                ],
            ),
        )
        return

    state_service.set_state(chat_session_id, state)

    agent_names = ", ".join(AGENT_DISPLAY_NAMES[a] for a in agents_to_call)
    status_msg = f"🏥 **Consulting specialist agents in parallel...**\n\n{agent_names}\n\n*Analyzing your query — full report coming shortly.*"
    await ctx.send(
        sender,
        ChatMessage(
            timestamp=datetime.now(tz=timezone.utc),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=status_msg)],
        ),
    )

    for agent_type in agents_to_call:
        await ctx.send(AGENT_ADDRESS_MAP[agent_type], state)


@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass
