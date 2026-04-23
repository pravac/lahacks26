import asyncio
import re
from agents.models.config import INSURANCE_NAVIGATOR_SEED
from agents.models.models import AgentResponse, MedicalAgentState
from agents.services.agent_runner import run_with_tools, build_query
from agents.services.tools import (
    SEARCH_INSURANCE_COVERAGE, SEARCH_WEB,
    search_npi_doctors, search_google_places_doctors,
)
from uagents import Agent, Context

harbor = Agent(
    name="harbor",
    seed=INSURANCE_NAVIGATOR_SEED,
    port=8005,
    endpoint=["http://127.0.0.1:8005/submit"],
)

SPECIALTY_KEYWORDS = {
    "cardiologist": "Cardiology",
    "cardiology": "Cardiology",
    "neurologist": "Neurology",
    "neurology": "Neurology",
    "oncologist": "Oncology",
    "oncology": "Oncology",
    "orthopedic": "Orthopedic Surgery",
    "psychiatrist": "Psychiatry",
    "psychiatry": "Psychiatry",
    "dermatologist": "Dermatology",
    "dermatology": "Dermatology",
    "endocrinologist": "Endocrinology",
    "gastroenterologist": "Gastroenterology",
    "rheumatologist": "Rheumatology",
    "urologist": "Urology",
    "pulmonologist": "Pulmonary Disease",
    "nephrologist": "Nephrology",
    "hematologist": "Hematology",
}

SYSTEM_PROMPT = """You are Harbor, an expert healthcare insurance navigator and patient advocate.

The user message may include pre-fetched NPI registry data (doctor listings) at the top — treat that as real, verified data and include it verbatim in your response under "Nearby Specialists".

Use your tools:
- Use search_insurance_coverage to search CMS, healthcare.gov, and Medicare/Medicaid official sources
- Use search_web for additional patient rights or billing information

Provide:
1. **Nearby Specialists** — You MUST list specific doctor names. Extract every name from the pre-fetched NPI and web data and list them with address and phone. NEVER tell the user to "check a website" or "search online" — give them the actual names. If a name appears in the data, include it.
2. **Direct Answer** — Address their insurance question with information from official sources
3. **What To Do Next** — Step-by-step actions (who to call, what to say, what to document)
4. **Know Your Rights** — Relevant patient protections with sources (cms.gov, healthcare.gov)
5. **Watch Out For** — Common insurer tactics or gotchas

If a denial sounds wrongful, say so directly and explain how to fight it.
Always cite sources."""


def _extract_zip(text: str) -> str | None:
    match = re.search(r"\b(\d{5})\b", text)
    return match.group(1) if match else None


def _extract_specialty(text: str) -> str | None:
    lower = text.lower()
    for keyword, specialty in SPECIALTY_KEYWORDS.items():
        if keyword in lower:
            return specialty
    return None


@harbor.on_message(MedicalAgentState)
async def handle_message(ctx: Context, sender: str, state: MedicalAgentState):
    ctx.logger.info(f"Harbor navigating insurance for session={state.chat_session_id}")

    query = state.query
    zip_code = _extract_zip(query)
    specialty = _extract_specialty(query)

    doctor_prefix = ""
    if zip_code and specialty:
        ctx.logger.info(f"Harbor pre-fetching doctors: specialty={specialty}, zip={zip_code}")
        npi_result, places_result = await asyncio.gather(
            search_npi_doctors(specialty=specialty, zip_code=zip_code),
            search_google_places_doctors(specialty=specialty, location=zip_code),
        )
        doctor_prefix = (
            f"[NPI Registry — Licensed {specialty} providers near {zip_code}]\n{npi_result}\n\n"
            f"[Google Maps — Rated {specialty} providers near {zip_code}]\n{places_result}\n\n"
        )

    enriched_query = doctor_prefix + build_query(state)

    result = await run_with_tools(
        query=enriched_query if enriched_query else build_query(state),
        system_prompt=SYSTEM_PROMPT,
        tools=[SEARCH_INSURANCE_COVERAGE, SEARCH_WEB],
    )
    ctx.logger.info(f"Harbor complete for session={state.chat_session_id}")
    await ctx.send(sender, AgentResponse(
        chat_session_id=state.chat_session_id,
        agent_type="insurance",
        result=result,
    ))


if __name__ == "__main__":
    harbor.run()
