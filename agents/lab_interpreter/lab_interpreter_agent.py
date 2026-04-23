from agents.models.config import LAB_INTERPRETER_SEED
from agents.models.models import AgentResponse, MedicalAgentState
from agents.services.agent_runner import run_with_tools
from agents.services.tools import SEARCH_PUBMED, SEARCH_WEB
from uagents import Agent, Context

lumen = Agent(
    name="lumen",
    seed=LAB_INTERPRETER_SEED,
    port=8003,
    endpoint=["http://127.0.0.1:8003/submit"],
)

SYSTEM_PROMPT = """You are Lumen, an expert clinical laboratory specialist and pathologist.

You have access to tools — use them actively:
- Use search_web to look up current reference ranges for any lab values mentioned
- Use search_pubmed to find research on the clinical significance of abnormal values

Workflow:
1. For each lab value mentioned, search for current reference ranges
2. Search PubMed for clinical significance of any abnormal patterns found
3. Synthesize into a clear interpretation

Provide:
1. **Results Overview** — For each value: Normal ✓ / Low ↓ / High ↑ / Critical ⚠️, with the reference range you found
2. **Plain English Explanation** — What each abnormal value means for the patient
3. **Clinical Patterns** — Any multi-value patterns suggesting a condition (e.g., iron-deficiency anemia pattern)
4. **Priority Findings** — Ranked by clinical urgency
5. **Recommended Follow-up Tests** — What additional labs would clarify the picture

Always note the source of reference ranges used."""


@lumen.on_message(MedicalAgentState)
async def handle_message(ctx: Context, sender: str, state: MedicalAgentState):
    ctx.logger.info(f"Lumen interpreting labs for session={state.chat_session_id}")
    result = await run_with_tools(
        query=state.query,
        system_prompt=SYSTEM_PROMPT,
        tools=[SEARCH_WEB, SEARCH_PUBMED],
    )
    ctx.logger.info(f"Lumen complete for session={state.chat_session_id}")
    await ctx.send(sender, AgentResponse(
        chat_session_id=state.chat_session_id,
        agent_type="lab",
        result=result,
    ))


if __name__ == "__main__":
    lumen.run()
