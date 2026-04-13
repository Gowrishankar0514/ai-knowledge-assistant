from langchain.memory import ConversationBufferMemory
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ChatMemoryManager:
    """Manages multi-turn conversation memory for the chatbot."""
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer" # Ensure the LLM output binds here
        )
        logger.info("Initialized ConversationBufferMemory.")

    def get_memory(self):
        return self.memory

    def clear(self):
        self.memory.clear()
        logger.info("Chat memory cleared.")
