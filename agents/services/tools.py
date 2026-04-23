import asyncio
import os
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

SEARCH_CLINICAL_TRIALS = {
    "type": "function",
    "function": {
        "name": "search_clinical_trials",
        "description": "Search ClinicalTrials.gov for active recruiting trials relevant to a medical condition.",
        "parameters": {
            "type": "object",
            "properties": {
                "condition": {"type": "string", "description": "Medical condition or diagnosis to search trials for"},
            },
            "required": ["condition"],
        },
    },
}

SEARCH_NPI_DOCTORS = {
    "type": "function",
    "function": {
        "name": "search_npi_doctors",
        "description": "Search the federal NPI registry for real licensed doctors or specialists near a zip code.",
        "parameters": {
            "type": "object",
            "properties": {
                "specialty": {"type": "string", "description": "Medical specialty (e.g. Cardiology, Hematology, General Practice)"},
                "zip_code": {"type": "string", "description": "US zip code to search near"},
            },
            "required": ["specialty", "zip_code"],
        },
    },
}


SEND_EMERGENCY_SMS = {
    "type": "function",
    "function": {
        "name": "send_emergency_sms",
        "description": "Send an emergency SMS alert to the patient's designated emergency contact. Use ONLY when urgency is CRITICAL (life-threatening situation).",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The emergency alert message to send"},
            },
            "required": ["message"],
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
            reactions = [r.get("reactionmeddrapt", "unknown") for r in report.get("patient", {}).get("reaction", [])]
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


async def search_clinical_trials(condition: str) -> str:
    try:
        url = (
            f"https://clinicaltrials.gov/api/v2/studies"
            f"?query.cond={condition.replace(' ', '+')}"
            f"&filter.overallStatus=RECRUITING"
            f"&pageSize=4"
            f"&fields=NCTId,BriefTitle,Phase,LocationCity,LocationState,EligibilityCriteria"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status != 200:
                    return f"No clinical trials found for '{condition}'."
                data = await r.json()

        studies = data.get("studies", [])
        if not studies:
            return f"No active recruiting trials found for '{condition}'."

        total = data.get("totalCount", len(studies))
        parts = [f"ClinicalTrials.gov — {total} recruiting trials found for '{condition}':\n"]
        for s in studies[:4]:
            proto = s.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design_mod = proto.get("designModule", {})
            nct_id = id_mod.get("nctId", "")
            title = id_mod.get("briefTitle", "")
            phase = ", ".join(design_mod.get("phases", ["N/A"]))
            parts.append(f"• **{title}** ({phase})\n  ID: {nct_id} | https://clinicaltrials.gov/study/{nct_id}")
        return "\n".join(parts)
    except Exception as e:
        return f"ClinicalTrials.gov search unavailable: {e}"


async def search_npi_doctors(specialty: str, zip_code: str) -> str:
    try:
        url = (
            f"https://npiregistry.cms.hhs.gov/api/"
            f"?version=2.1"
            f"&taxonomy_description={specialty.replace(' ', '+')}"
            f"&postal_code={zip_code}"
            f"&limit=5"
            f"&enumeration_type=NPI-1"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as r:
                if r.status != 200:
                    return f"NPI registry search unavailable."
                data = await r.json()

        results = data.get("results", [])
        if not results:
            return f"No {specialty} providers found near zip code {zip_code} in the NPI registry."

        count = data.get("result_count", len(results))
        parts = [f"NPI Registry — {count} {specialty} providers near {zip_code}:\n"]
        for p in results[:5]:
            basic = p.get("basic", {})
            name = f"Dr. {basic.get('first_name', '')} {basic.get('last_name', '')}".strip()
            credential = basic.get("credential", "")
            if credential:
                name += f", {credential}"
            addresses = p.get("addresses", [{}])
            addr = addresses[0] if addresses else {}
            city = addr.get("city", "")
            state = addr.get("state", "")
            phone = addr.get("telephone_number", "")
            taxonomy = p.get("taxonomies", [{}])[0].get("desc", specialty)
            parts.append(f"• **{name}** — {taxonomy}\n  {city}, {state}  📞 {phone}")
        return "\n".join(parts)
    except Exception as e:
        return f"NPI registry search unavailable: {e}"



async def search_google_places_doctors(specialty: str, location: str) -> str:
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        return "Google Places API key not configured (GOOGLE_PLACES_API_KEY missing from .env)."
    try:
        url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": (
                "places.displayName,places.formattedAddress,"
                "places.internationalPhoneNumber,places.rating,"
                "places.userRatingCount,places.businessStatus"
            ),
        }
        body = {
            "textQuery": f"{specialty} near {location}",
            "maxResultCount": 8,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=8)) as r:
                if r.status != 200:
                    text = await r.text()
                    return f"Google Places search unavailable (status {r.status}): {text[:200]}"
                data = await r.json()
        places = data.get("places", [])
        if not places:
            return f"No Google Places results found for {specialty} near {location}."
        parts = [f"Google Maps — Top {specialty} providers near {location}:\n"]
        for p in places:
            name = p.get("displayName", {}).get("text", "")
            address = p.get("formattedAddress", "")
            phone = p.get("internationalPhoneNumber", "")
            rating = p.get("rating", "")
            reviews = p.get("userRatingCount", "")
            status = p.get("businessStatus", "")
            rating_str = f"⭐ {rating} ({reviews} reviews)" if rating else ""
            status_str = f" — {status.replace('_', ' ').title()}" if status else ""
            parts.append(f"• **{name}** {rating_str}{status_str}\n  {address}  📞 {phone}")
        return "\n".join(parts)
    except Exception as e:
        return f"Google Places search unavailable: {e}"


async def send_emergency_sms(message: str) -> str:
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    to_number = os.getenv("EMERGENCY_CONTACT_NUMBER")

    if not all([account_sid, auth_token, from_number, to_number]):
        return "⚠️ Emergency SMS not configured (missing Twilio credentials in .env). Alert would have been sent to emergency contact."

    try:
        from twilio.rest import Client
        client = Client(account_sid, auth_token)
        sms = client.messages.create(
            body=f"🚨 MEDAGENT EMERGENCY ALERT 🚨\n\n{message}\n\nThis is an automated alert from MedAgent.",
            from_=from_number,
            to=to_number,
        )
        return f"✅ Emergency SMS sent to emergency contact (SID: {sms.sid})"
    except Exception as e:
        return f"Emergency SMS failed: {e}"


# ---------------------------------------------------------------------------
# Handler map — passed to agent_runner
# ---------------------------------------------------------------------------

ALL_TOOL_HANDLERS = {
    "search_web": search_web,
    "search_pubmed": search_pubmed,
    "lookup_fda_drug": lookup_fda_drug,
    "check_fda_drug_events": check_fda_drug_events,
    "search_insurance_coverage": search_insurance_coverage,
    "search_clinical_trials": search_clinical_trials,
    "search_npi_doctors": search_npi_doctors,
    "search_google_places_doctors": search_google_places_doctors,
    "send_emergency_sms": send_emergency_sms,
}
