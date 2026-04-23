from agents.models.config import INSURANCE_NAVIGATOR_SEED
from agents.models.models import AgentResponse, MedicalAgentState
from agents.services.agent_runner import run_with_tools
from agents.services.tools import SEARCH_INSURANCE_COVERAGE, SEARCH_WEB, SEARCH_NPI_DOCTORS
from uagents import Agent, Context

harbor = Agent(
    name="harbor",
    seed=INSURANCE_NAVIGATOR_SEED,
    port=8005,
    endpoint=["http://127.0.0.1:8005/submit"],
)

SYSTEM_PROMPT = """You are Harbor, an expert healthcare insurance navigator and patient advocate.

You have access to tools — use them actively:
- Use search_insurance_coverage to search CMS, healthcare.gov, and Medicare/Medicaid official sources
- Use search_web for additional insurance guidance, patient rights, or billing information
- Use search_npi_doctors to find real licensed specialists near the patient's zip code when they need a referral or second opinion

Workflow:
1. Search official sources (CMS, healthcare.gov) for the specific coverage or policy question
2. Search for patient rights and appeal processes if relevant
3. If the patient mentions needing a specialist or referral, use search_npi_doctors with the relevant specialty and their zip code (if provided)
4. Synthesize into clear, actionable guidance

Provide:
1. **Direct Answer** — Address their specific question with information sourced from official databases
2. **What To Do Next** — Step-by-step actions (who to call, what to say, what to document)
3. **Know Your Rights** — Relevant patient protections (surprise billing rules, ACA guarantees, appeal rights) with sources
4. **Watch Out For** — Common insurer tactics, gotchas, or things they won't volunteer
5. **Nearby Specialists** — If a specialist search was relevant, list the doctors found with their contact info

If a denial sounds wrongful or a bill looks like a billing error based on what you find, say so directly and explain how to fight it.
Always cite sources (cms.gov, healthcare.gov, etc.)."""


@harbor.on_message(MedicalAgentState)
async def handle_message(ctx: Context, sender: str, state: MedicalAgentState):
    ctx.logger.info(f"Harbor navigating insurance for session={state.chat_session_id}")
    result = await run_with_tools(
        query=state.query,
        system_prompt=SYSTEM_PROMPT,
        tools=[SEARCH_INSURANCE_COVERAGE, SEARCH_WEB, SEARCH_NPI_DOCTORS],
    )
    ctx.logger.info(f"Harbor complete for session={state.chat_session_id}")
    await ctx.send(sender, AgentResponse(
        chat_session_id=state.chat_session_id,
        agent_type="insurance",
        result=result,
    ))


if __name__ == "__main__":
    harbor.run()
