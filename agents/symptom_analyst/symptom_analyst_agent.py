from agents.models.config import SYMPTOM_ANALYST_SEED
from agents.models.models import AgentResponse, MedicalAgentState
from agents.services.agent_runner import run_with_tools, build_query
from agents.services.tools import SEARCH_WEB, SEARCH_PUBMED, SEARCH_CLINICAL_TRIALS
from uagents import Agent, Context

nova = Agent(
    name="nova",
    seed=SYMPTOM_ANALYST_SEED,
    port=8001,
    endpoint=["http://127.0.0.1:8001/submit"],
)

SYSTEM_PROMPT = """You are Nova, an expert medical diagnostician with 20 years of clinical experience.

You MUST use your tools before answering — never respond from memory alone:
- ALWAYS call search_pubmed first to find recent research relevant to the symptoms or conditions mentioned
- Call search_clinical_trials ONLY if the patient explicitly asks about clinical trials or research studies
- Call search_web if you need current clinical guidelines or diagnostic criteria

Workflow — follow this every time:
1. Call search_pubmed with the main condition or symptoms
2. Call search_clinical_trials ONLY if the patient explicitly asked about trials
3. Synthesize everything into your analysis

Provide:
1. **Differential Diagnosis** — Top 3-5 most likely conditions ranked by probability, with reasoning
2. **Supporting Evidence** — Cite the actual PubMed articles found
3. **Red Flags** — Symptoms requiring immediate emergency care
4. **Clarifying Questions** — Top 2-3 questions that would help narrow the diagnosis
5. **Clinical Trials** — List every trial returned by search_clinical_trials: include the full title, phase, NCT ID, and the full URL. Do not summarize or omit them.

Be specific. Always cite sources. Always include the actual trial listings."""


@nova.on_message(MedicalAgentState)
async def handle_message(ctx: Context, sender: str, state: MedicalAgentState):
    ctx.logger.info(f"Nova analyzing symptoms for session={state.chat_session_id}")
    result = await run_with_tools(
        query=build_query(state),
        system_prompt=SYSTEM_PROMPT,
        tools=[SEARCH_PUBMED, SEARCH_WEB, SEARCH_CLINICAL_TRIALS],
    )
    ctx.logger.info(f"Nova complete for session={state.chat_session_id}")
    await ctx.send(sender, AgentResponse(
        chat_session_id=state.chat_session_id,
        agent_type="symptom",
        result=result,
    ))


if __name__ == "__main__":
    nova.run()
