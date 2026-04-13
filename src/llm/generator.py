from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from src.config import Config
from src.utils.logger import get_logger
from src.llm.prompts import PERSONAS
from src.llm.memory import ChatMemoryManager

logger = get_logger(__name__)

class RAGGenerator:
    def __init__(self, retriever, memory_manager: ChatMemoryManager, persona: str = "Baseline Q&A"):
        self.retriever = retriever
        self.memory_manager = memory_manager
        self.memory = memory_manager.get_memory()
        self.current_persona = persona
        
        # Decide which LLM to use based on configuration
        open_ai_key = Config.OPENAI_API_KEY
        if open_ai_key and open_ai_key.strip() != "" and open_ai_key != "your_openai_api_key_here":
            logger.info("Initializing OpenAI GPT-3.5-turbo model.")
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
        else:
            logger.info("OpenAI key not found. Initializing HuggingFace model.")
            self.llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                temperature=0.3,
                huggingfacehub_api_token=Config.HUGGINGFACEHUB_API_TOKEN
            )

        self._build_chain()

    def set_persona(self, persona: str):
        if persona != self.current_persona and persona in PERSONAS:
            self.current_persona = persona
            logger.info(f"Switching AI Persona to: {persona}")
            self._build_chain()

    def _build_chain(self):
        prompt_text = PERSONAS.get(self.current_persona, PERSONAS["Baseline Q&A"])
        self.custom_prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["context", "chat_history", "question"]
        )

        if self.retriever and self.retriever.compression_retriever:
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever.compression_retriever,
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": self.custom_prompt},
                return_source_documents=True,
                verbose=False
            )
        else:
            self.qa_chain = None

    def query(self, user_question: str):
        if not self.qa_chain:
            logger.error("QA chain is not initialized due to missing retriever.")
            return {"answer": "System is not initialized properly. Please ingest data first.", "source_documents": []}
            
        logger.info(f"Generating answer for query: '{user_question}' under persona '{self.current_persona}'")
        try:
            response = self.qa_chain.invoke({"question": user_question})
            return response
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {"answer": f"An error occurred while generating the answer: {str(e)}", "source_documents": []}
