import asyncio
import aiohttp

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

SEARCH_WEB = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for current medical information, guidelines, or health-related content.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
}

SEARCH_PUBMED = {
    "type": "function",
    "function": {
        "name": "search_pubmed",
        "description": "Search PubMed for peer-reviewed medical research articles relevant to a clinical question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Clinical search query"},
            },
            "required": ["query"],
        },
    },
}

LOOKUP_FDA_DRUG = {
    "type": "function",
    "function": {
        "name": "lookup_fda_drug",
        "description": "Look up official FDA drug label data including warnings, contraindications, and interactions.",
        "parameters": {
            "type": "object",
            "properties": {
                "drug_name": {"type": "string", "description": "Brand or generic drug name"},
            },
            "required": ["drug_name"],
        },
    },
}

CHECK_FDA_DRUG_EVENTS = {
    "type": "function",
    "function": {
        "name": "check_fda_drug_events",
        "description": "Search the FDA adverse event reporting system (FAERS) for safety signals when multiple drugs are taken together.",
        "parameters": {
            "type": "object",
            "properties": {
                "drugs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of drug names to check for co-prescription adverse events",
                },
            },
            "required": ["drugs"],
        },
    },
}

SEARCH_INSURANCE_COVERAGE = {
    "type": "function",
    "function": {
        "name": "search_insurance_coverage",
        "description": "Search CMS, healthcare.gov, and official sources for insurance coverage, billing, and patient rights information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Insurance or coverage question"},
            },
            "required": ["query"],
        },
    },
}

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def search_web(query: str) -> str:
    try:
        from ddgs import DDGS
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, lambda: list(DDGS().text(query, max_results=4))
        )
        if not results:
            return "No web results found."
        parts = [f"**{r['title']}**\n{r['body']}" for r in results]
        return "\n\n".join(parts)
    except Exception as e:
        return f"Web search unavailable: {e}"


async def search_pubmed(query: str) -> str:
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        async with aiohttp.ClientSession() as session:
            search_url = f"{base}/esearch.fcgi?db=pubmed&term={query}&retmax=4&retmode=json"
            async with session.get(search_url, timeout=aiohttp.ClientTimeout(total=8)) as r:
                data = await r.json()
            ids = data.get("esearchresult", {}).get("idlist", [])
            if not ids:
                return "No PubMed results found."
            ids_str = ",".join(ids)
            summary_url = f"{base}/esummary.fcgi?db=pubmed&id={ids_str}&retmode=json"
            async with session.get(summary_url, timeout=aiohttp.ClientTimeout(total=8)) as r:
                data = await r.json()
        results = []
        for uid, article in data.get("result", {}).items():
            if uid == "uids":
                continue
            title = article.get("title", "")
            source = article.get("source", "")
            pubdate = article.get("pubdate", "")
            results.append(f"- {title} ({source}, {pubdate})")
        return "PubMed articles:\n" + "\n".join(results)
    except Exception as e:
        return f"PubMed search unavailable: {e}"


async def lookup_fda_drug(drug_name: str) -> str:
    try:
        url = (
            f"https://api.fda.gov/drug/label.json"
            f"?search=openfda.brand_name:\"{drug_name}\"+OR+openfda.generic_name:\"{drug_name}\""
            f"&limit=1"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as r:
                if r.status != 200:
                    return f"No FDA label found for '{drug_name}'."
                data = await r.json()
        label = data.get("results", [{}])[0]
        sections = []
        for field, heading in [
            ("warnings", "WARNINGS"),
            ("drug_interactions", "DRUG INTERACTIONS"),
            ("contraindications", "CONTRAINDICATIONS"),
            ("boxed_warning", "BOXED WARNING"),
        ]:
            if field in label:
                text = " ".join(label[field])[:600]
                sections.append(f"{heading}: {text}")
        if not sections:
            return f"FDA label found for '{drug_name}' but contains no warnings or interaction data."
        return f"FDA Drug Label — {drug_name}:\n\n" + "\n\n".join(sections)
    except Exception as e:
        return f"FDA lookup unavailable: {e}"


async def check_fda_drug_events(drugs: list) -> str:
    try:
        # Search FDA FAERS for adverse events involving all listed drugs together
        drug_terms = "+AND+".join(
            f'patient.drug.medicinalproduct:"{d}"' for d in drugs
        )
        url = f"https://api.fda.gov/drug/event.json?search={drug_terms}&limit=5"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status != 200:
                    return f"No FDA adverse event data found for combination: {', '.join(drugs)}."
                data = await r.json()

        results = data.get("results", [])
        if not results:
            return f"No adverse event reports found in FAERS for {', '.join(drugs)} together."

        total = data.get("meta", {}).get("results", {}).get("total", len(results))
        summary_parts = [f"FDA FAERS found {total} adverse event report(s) for {', '.join(drugs)} used together.\n"]

        for i, report in enumerate(results[:3], 1):
            reactions = [
                r.get("reactionmeddrapt", "unknown")
                for r in report.get("patient", {}).get("reaction", [])
            ]
            serious = report.get("serious", "")
            seriousness = []
            if report.get("seriousnessdeath") == "1":
                seriousness.append("death")
            if report.get("seriousnesshospitalization") == "1":
                seriousness.append("hospitalization")
            if report.get("seriousnesslifethreatening") == "1":
                seriousness.append("life-threatening")

            summary_parts.append(
                f"Report {i}: Reactions — {', '.join(reactions[:5])}."
                + (f" Serious outcomes: {', '.join(seriousness)}." if seriousness else "")
            )

        return "\n".join(summary_parts)
    except Exception as e:
        return f"FDA adverse event lookup unavailable: {e}"


async def search_insurance_coverage(query: str) -> str:
    targeted_query = f"{query} site:cms.gov OR site:healthcare.gov OR site:medicare.gov"
    return await search_web(targeted_query)


# ---------------------------------------------------------------------------
# Handler map — passed to agent_runner
# ---------------------------------------------------------------------------

ALL_TOOL_HANDLERS = {
    "search_web": search_web,
    "search_pubmed": search_pubmed,
    "lookup_fda_drug": lookup_fda_drug,
    "check_fda_drug_events": check_fda_drug_events,
    "search_insurance_coverage": search_insurance_coverage,
}
