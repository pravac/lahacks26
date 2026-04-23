import os
from dotenv import find_dotenv, load_dotenv
from uagents_core.identity import Identity

load_dotenv(find_dotenv())

ORCHESTRATOR_SEED = os.getenv("ORCHESTRATOR_SEED_PHRASE")
SYMPTOM_ANALYST_SEED = os.getenv("SYMPTOM_ANALYST_SEED_PHRASE")
DRUG_INTERACTION_SEED = os.getenv("DRUG_INTERACTION_SEED_PHRASE")
LAB_INTERPRETER_SEED = os.getenv("LAB_INTERPRETER_SEED_PHRASE")
RISK_ASSESSOR_SEED = os.getenv("RISK_ASSESSOR_SEED_PHRASE")
INSURANCE_NAVIGATOR_SEED = os.getenv("INSURANCE_NAVIGATOR_SEED_PHRASE")

SYMPTOM_ANALYST_ADDRESS = Identity.from_seed(seed=SYMPTOM_ANALYST_SEED, index=0).address
DRUG_INTERACTION_ADDRESS = Identity.from_seed(seed=DRUG_INTERACTION_SEED, index=0).address
LAB_INTERPRETER_ADDRESS = Identity.from_seed(seed=LAB_INTERPRETER_SEED, index=0).address
RISK_ASSESSOR_ADDRESS = Identity.from_seed(seed=RISK_ASSESSOR_SEED, index=0).address
INSURANCE_NAVIGATOR_ADDRESS = Identity.from_seed(seed=INSURANCE_NAVIGATOR_SEED, index=0).address
