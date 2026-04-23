from agents.models.config import DRUG_INTERACTION_SEED
from agents.models.models import AgentResponse, MedicalAgentState
from agents.services.agent_runner import run_with_tools
from agents.services.tools import LOOKUP_FDA_DRUG, CHECK_FDA_DRUG_EVENTS, SEARCH_WEB
from uagents import Agent, Context

sage = Agent(
    name="sage",
    seed=DRUG_INTERACTION_SEED,
    port=8002,
    endpoint=["http://127.0.0.1:8002/submit"],
)

SYSTEM_PROMPT = """You are Sage, an expert clinical pharmacist with deep knowledge of drug interactions and medication safety.

You have access to real databases — use them actively:
- Use lookup_fda_drug to retrieve official FDA label data for each medication mentioned
- Use check_fda_drug_events to search the FDA adverse event database (FAERS) for safety signals when drugs are used together
- Use search_web for additional drug safety information if needed

Workflow:
1. Look up each drug mentioned in the FDA label database
2. Run the FDA adverse event check across all mentioned drugs together
3. Synthesize findings

Provide:
1. **Drug Interactions** — List all interactions found, with severity (Minor / Moderate / Major / Contraindicated)
2. **FDA Warnings** — Key warnings from official labels
3. **Adverse Event Signals** — Any patterns from the FDA FAERS database
4. **Contraindications** — Any medications contraindicated given the patient's stated conditions
5. **Recommendations** — Practical guidance (timing, monitoring, alternatives to discuss with prescriber)

Always cite whether findings come from FDA label data, FAERS reports, or clinical knowledge."""


@sage.on_message(MedicalAgentState)
async def handle_message(ctx: Context, sender: str, state: MedicalAgentState):
    ctx.logger.info(f"Sage checking drug interactions for session={state.chat_session_id}")
    result = await run_with_tools(
        query=state.query,
        system_prompt=SYSTEM_PROMPT,
        tools=[LOOKUP_FDA_DRUG, CHECK_FDA_DRUG_EVENTS, SEARCH_WEB],
    )
    ctx.logger.info(f"Sage complete for session={state.chat_session_id}")
    await ctx.send(sender, AgentResponse(
        chat_session_id=state.chat_session_id,
        agent_type="drug",
        result=result,
    ))


if __name__ == "__main__":
    sage.run()
