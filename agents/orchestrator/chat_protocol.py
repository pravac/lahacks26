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
- If the query is NOT medical or health-related at all (greetings, agent identity, small talk) → return agents: []

Respond with ONLY a valid JSON object with these fields:
- "agents": array of agent names to consult (or [] for non-medical)
- "suspected_conditions": brief comma-separated list of conditions hinted at by the query (empty string if non-medical or unclear)
- "clarifying_question": one short follow-up question that would most help the specialists (empty string if enough info or non-medical)

Examples:
- Symptoms only: {"agents": ["symptom", "risk"], "suspected_conditions": "anemia, iron deficiency", "clarifying_question": "How long have you been experiencing these symptoms?"}
- Insurance only: {"agents": ["insurance"], "suspected_conditions": "", "clarifying_question": "What type of insurance plan do you have (HMO, PPO, etc.)?"}
- Both: {"agents": ["symptom", "drug", "risk", "insurance"], "suspected_conditions": "cardiac event", "clarifying_question": "Are you experiencing chest pain right now?"}
- Non-medical: {"agents": [], "suspected_conditions": "", "clarifying_question": ""}"""

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


EMERGENCY_TRIAGE_PROMPT = """You are an emergency triage AI. Respond with only "YES" or "NO".

Is the following message describing a situation where someone needs 911 called RIGHT NOW — meaning there is an active life-threatening emergency, severe physical trauma, or immediate danger to life?

Examples of YES: heart attack, stroke, can't breathe, car accident with injury, run over, broken bones, severe bleeding, overdose, choking, drowning, anaphylaxis, suicide attempt, fire/burn injury, electrocution.
Examples of NO: chronic illness questions, medication questions, lab results, insurance questions, mild symptoms that have been ongoing."""

EMERGENCY_RESPONSE = """🚨 **THIS IS A MEDICAL EMERGENCY — CALL 911 NOW** 🚨

Your symptoms suggest a potentially life-threatening emergency.

**Do this immediately:**
1. **Call 911** (or have someone nearby call for you)
2. **Do not drive yourself** to the hospital
3. **Chew an aspirin** (325mg) if available and you are not allergic
4. **Sit or lie down** and stay as calm as possible
5. **Unlock your front door** so paramedics can enter

Your emergency contact has been notified.

---
⚠️ *Do not wait for an AI response. Call 911 now.*"""


async def _is_emergency(query: str) -> bool:
    client = get_llm_client()
    try:
        response = await client.chat.completions.create(
            model=ASI_MODEL,
            messages=[
                {"role": "system", "content": EMERGENCY_TRIAGE_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=5,
        )
        answer = response.choices[0].message.content.strip().upper()
        return answer.startswith("YES")
    except Exception:
        return False


async def decide_agents(query: str) -> dict:
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
        parsed = json.loads(content.strip())
        agents = parsed.get("agents", [])
        if agents and "risk" not in agents:
            agents.append("risk")
        valid = [a for a in agents if a in AGENT_ADDRESS_MAP]
        return {
            "agents": valid[:3],
            "suspected_conditions": parsed.get("suspected_conditions", ""),
            "clarifying_question": parsed.get("clarifying_question", ""),
        }
    except Exception:
        return {"agents": ["symptom", "risk"], "suspected_conditions": "", "clarifying_question": ""}


@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.now(), acknowledged_msg_id=msg.msg_id),
    )

    text = " ".join(item.text for item in msg.content if isinstance(item, TextContent))
    ctx.logger.info(f"Received query: {text!r}")

    # Emergency fast-path — LLM triage call (fast, single token response)
    is_emergency = await _is_emergency(text)
    if is_emergency:
        ctx.logger.info("EMERGENCY detected — sending immediate response")
        await ctx.send(
            sender,
            ChatMessage(
                timestamp=datetime.now(tz=timezone.utc),
                msg_id=uuid4(),
                content=[
                    TextContent(type="text", text=EMERGENCY_RESPONSE),
                    EndSessionContent(type="end-session"),
                ],
            ),
        )
        # Fire Sentinel in background to send the emergency SMS
        from agents.services.tools import send_emergency_sms
        await send_emergency_sms(f"Emergency query received: {text[:300]}")
        return

    chat_session_id = str(ctx.session)

    routing = await decide_agents(text)
    agents_to_call = routing["agents"]
    suspected_conditions = routing["suspected_conditions"]
    clarifying_question = routing["clarifying_question"]
    ctx.logger.info(f"Routing to agents: {agents_to_call}")

    state = MedicalAgentState(
        chat_session_id=chat_session_id,
        query=text,
        user_sender_address=sender,
        agents_to_call=agents_to_call,
        suspected_conditions=suspected_conditions,
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
    status_parts = [f"🏥 **Consulting specialist agents in parallel...**\n\n{agent_names}"]
    if clarifying_question:
        status_parts.append(f"\n💬 **While you wait:** {clarifying_question}")
    status_parts.append("\n*Full report coming shortly.*")
    status_msg = "\n".join(status_parts)
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
