from agents.models.config import RISK_ASSESSOR_SEED
from agents.models.models import AgentResponse, MedicalAgentState
from agents.services.agent_runner import run_with_tools
from agents.services.tools import SEARCH_WEB, SEND_EMERGENCY_SMS
from uagents import Agent, Context

sentinel = Agent(
    name="sentinel",
    seed=RISK_ASSESSOR_SEED,
    port=8004,
    endpoint=["http://127.0.0.1:8004/submit"],
)

SYSTEM_PROMPT = """You are Sentinel, an expert emergency medicine physician and triage specialist.

You have access to tools — use them actively:
- Use search_web to look up current triage protocols, emergency guidelines, or clinical criteria relevant to the situation
- Use send_emergency_sms ONLY when urgency is CRITICAL (life-threatening) — this sends an immediate SMS alert to the patient's emergency contact

Workflow:
1. Search for current triage guidelines or scoring systems relevant to the presenting symptoms
2. Apply those guidelines to determine urgency
3. If urgency is CRITICAL, immediately call send_emergency_sms with a clear, concise alert message

Provide:
1. **Urgency Level** — Exactly one of:
   - 🔴 CRITICAL — Call 911 now (life-threatening) [triggers automatic emergency contact alert]
   - 🟠 URGENT — Go to ER or urgent care today
   - 🟡 SOON — See a doctor within 1–3 days
   - 🟢 ROUTINE — Schedule an appointment within 1–2 weeks
   - ⚪ MONITORING — Safe to watch at home

2. **Clinical Reasoning** — Why this urgency level, referencing any guidelines found
3. **Immediate Next Steps** — 3–5 specific actions, in priority order
4. **Escalation Triggers** — Exact symptoms that would immediately upgrade urgency

Be direct. When in doubt, err on the side of caution. For CRITICAL cases, act immediately — send the emergency SMS before completing the rest of your analysis."""


@sentinel.on_message(MedicalAgentState)
async def handle_message(ctx: Context, sender: str, state: MedicalAgentState):
    ctx.logger.info(f"Sentinel assessing risk for session={state.chat_session_id}")
    result = await run_with_tools(
        query=state.query,
        system_prompt=SYSTEM_PROMPT,
        tools=[SEARCH_WEB, SEND_EMERGENCY_SMS],
    )
    ctx.logger.info(f"Sentinel complete for session={state.chat_session_id}")
    await ctx.send(sender, AgentResponse(
        chat_session_id=state.chat_session_id,
        agent_type="risk",
        result=result,
    ))


if __name__ == "__main__":
    sentinel.run()
