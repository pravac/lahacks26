from agents.models.models import MedicalAgentState


class InMemoryStateService:
    def __init__(self) -> None:
        self._store: dict[str, MedicalAgentState] = {}

    def set_state(self, chat_session_id: str, state: MedicalAgentState) -> None:
        self._store[chat_session_id] = state

    def get_state(self, chat_session_id: str) -> MedicalAgentState | None:
        return self._store.get(chat_session_id)


state_service = InMemoryStateService()
