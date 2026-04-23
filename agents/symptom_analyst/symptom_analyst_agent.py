from agents.models.config import SYMPTOM_ANALYST_SEED
from agents.models.models import AgentResponse, MedicalAgentState
from agents.services.agent_runner import run_with_tools
from agents.services.tools import SEARCH_WEB, SEARCH_PUBMED
from uagents import Agent, Context

nova = Agent(
    name="nova",
    seed=SYMPTOM_ANALYST_SEED,
    port=8001,
    endpoint=["http://127.0.0.1:8001/submit"],
)

SYSTEM_PROMPT = """You are Nova, an expert medical diagnostician with 20 years of clinical experience.

You have access to tools — use them actively:
- Use search_pubmed to find recent research on conditions matching the symptoms
- Use search_web to look up current clinical guidelines or diagnostic criteria

Workflow:
1. Search PubMed for relevant conditions
2. Search for current diagnostic guidelines if needed
3. Synthesize findings into your analysis

Provide:
1. **Differential Diagnosis** — Top 3-5 most likely conditions ranked by probability, with reasoning
2. **Supporting Evidence** — Reference any articles or guidelines you found
3. **Red Flags** — Symptoms requiring immediate emergency care
4. **Clarifying Questions** — Top 2-3 questions that would help narrow the diagnosis

Be specific and cite your sources."""


@nova.on_message(MedicalAgentState)
async def handle_message(ctx: Context, sender: str, state: MedicalAgentState):
    ctx.logger.info(f"Nova analyzing symptoms for session={state.chat_session_id}")
    result = await run_with_tools(
        query=state.query,
        system_prompt=SYSTEM_PROMPT,
        tools=[SEARCH_PUBMED, SEARCH_WEB],
    )
    ctx.logger.info(f"Nova complete for session={state.chat_session_id}")
    await ctx.send(sender, AgentResponse(
        chat_session_id=state.chat_session_id,
        agent_type="symptom",
        result=result,
    ))


if __name__ == "__main__":
    nova.run()
