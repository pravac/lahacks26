orchestrator:
	python -m agents.orchestrator.orchestrator_fetchai_wrapped_agent

symptom-analyst:
	python -m agents.symptom_analyst.symptom_analyst_agent

drug-interaction:
	python -m agents.drug_interaction.drug_interaction_agent

lab-interpreter:
	python -m agents.lab_interpreter.lab_interpreter_agent

risk-assessor:
	python -m agents.risk_assessor.risk_assessor_agent

insurance-navigator:
	python -m agents.insurance_navigator.insurance_navigator_agent
