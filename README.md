# MedAgent

An AI-powered medical multi-agent system built on the [Fetch.ai](https://fetch.ai) uAgents framework. Describe your symptoms, medications, lab results, or insurance questions — MedAgent automatically routes your query to a team of specialist AI agents running in parallel, then synthesizes their findings into a single unified report.

## Architecture

```
User (ASI:One Chat)
        ↓
  Orchestrator
  LLM-driven routing — decides which agents to call
        ↓ (parallel dispatch)
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│     Nova     │     Sage     │    Lumen     │   Sentinel   │    Harbor    │
│  Symptom     │    Drug      │     Lab      │    Risk      │  Insurance   │
│  Analyst     │ Interaction  │ Interpreter  │  Assessor    │  Navigator   │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
        ↓ (results collected)
  Orchestrator synthesizes all findings
        ↓
  Unified clinical report → User
```

## Agents

| Agent | Role | Tools |
|---|---|---|
| **Nova** (Symptom Analyst) | Differential diagnosis | PubMed search, web search |
| **Sage** (Drug Interaction Checker) | Medication safety | FDA drug labels, FDA FAERS adverse event database |
| **Lumen** (Lab Interpreter) | Lab result interpretation | Web search for reference ranges, PubMed |
| **Sentinel** (Risk Assessor) | Urgency triage | Web search for clinical guidelines |
| **Harbor** (Insurance Navigator) | Insurance & billing guidance | CMS.gov, healthcare.gov search |

The orchestrator uses the ASI:One LLM to intelligently decide which subset of agents to invoke based on the query — not keyword matching. A query about symptoms and medications routes to Nova + Sage + Sentinel. A pure insurance question routes only to Harbor.

## Real Data Sources

Agents don't just use LLM knowledge — they actively query live databases:

- **PubMed** (NCBI E-utilities) — peer-reviewed medical literature
- **OpenFDA Drug Labels API** — official FDA drug label data including warnings and contraindications
- **FDA FAERS** — adverse event reporting database (15,000+ real reports for common drug combinations)
- **CMS / healthcare.gov** — Medicare, Medicaid, and insurance coverage information

## Setup

### 1. Clone and configure environment

```bash
cp .env.example .env
```

Fill in `.env` with random seed phrases and your ASI:One API key:

```
ORCHESTRATOR_SEED_PHRASE=<random string, no spaces>
SYMPTOM_ANALYST_SEED_PHRASE=<random string, no spaces>
DRUG_INTERACTION_SEED_PHRASE=<random string, no spaces>
LAB_INTERPRETER_SEED_PHRASE=<random string, no spaces>
RISK_ASSESSOR_SEED_PHRASE=<random string, no spaces>
INSURANCE_NAVIGATOR_SEED_PHRASE=<random string, no spaces>

ASI_ONE_API_KEY=<your key from asi1.ai>
```

Get your ASI:One API key at [asi1.ai](https://asi1.ai) → Developer settings.

### 2. Create virtual environment and install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Start all agents

Each agent runs in its own terminal:

```bash
make orchestrator       # port 8000
make symptom-analyst    # port 8001
make drug-interaction   # port 8002
make lab-interpreter    # port 8003
make risk-assessor      # port 8004
make insurance-navigator # port 8005
```

### 4. Connect the orchestrator mailbox

Open the orchestrator's Agent Inspector link (printed in the terminal on startup) and connect it to a mailbox on [agentverse.ai](https://agentverse.ai). The specialist agents communicate locally and don't need mailboxes.

### 5. Chat with MedAgent

Go to [asi1.ai](https://asi1.ai), find the orchestrator's agent profile, and click **Chat with Agent**.

## Example Queries

```
I've been exhausted and short of breath for 2 weeks, my ferritin came back at 6
and hemoglobin 8.2. I'm currently taking metformin and ibuprofen.
Will my insurance cover an iron infusion?
```

```
I have chest pain radiating to my left arm and I'm on warfarin and aspirin.
```

```
My insurance denied my MRI claim. What are my appeal rights?
```

```
My TSH is 8.2 and T4 is 0.7. What does this mean?
```

## Tech Stack

- [Fetch.ai uAgents](https://github.com/fetchai/uAgents) — agent framework and ASI:One chat protocol
- [ASI:One API](https://asi1.ai) — LLM powering all agent reasoning (OpenAI-compatible)
- [OpenFDA API](https://open.fda.gov) — drug labels and adverse event data
- [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/home/develop/api/) — PubMed literature search
- [ddgs](https://github.com/deedy5/ddgs) — web search
