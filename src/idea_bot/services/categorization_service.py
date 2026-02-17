from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI

from idea_bot.config import AppConfig


class Categorizer(ABC):
    @abstractmethod
    def categorize(self, title: str, description: str) -> str:
        raise NotImplementedError


class RuleBasedCategorizer(Categorizer):
    def categorize(self, title: str, description: str) -> str:
        text = f"{title}\n{description}".lower()
        if any(k in text for k in ["automation", "workflow", "efficiency"]):
            return "Process"
        if any(k in text for k in ["customer", "client", "support", "user"]):
            return "Customer"
        if any(k in text for k in ["revenue", "cost", "budget", "pricing"]):
            return "Finance"
        if any(k in text for k in ["hiring", "employee", "training"]):
            return "HR"
        if any(k in text for k in ["ai", "ml", "model", "data", "software", "app"]):
            return "Technology"
        return "Other"


class AzureLLMCategorizer(Categorizer):
    def __init__(self, llm: AzureChatOpenAI, fallback: Categorizer) -> None:
        self._llm = llm
        self._fallback = fallback

    def categorize(self, title: str, description: str) -> str:
        prompt = (
            "Categorize the following idea into one of: Product, Process, Customer, Operations, "
            "Technology, Finance, HR, Sustainability, Other. Return only the category name.\n\n"
            f"Idea:\n{title}\n{description}"
        )
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            candidate = response.content.strip().splitlines()[0]
            return candidate or self._fallback.categorize(title, description)
        except Exception:
            return self._fallback.categorize(title, description)


def build_categorizer(config: AppConfig) -> Categorizer:
    fallback = RuleBasedCategorizer()
    if config.azure_endpoint and config.azure_api_key and config.azure_chat_deployment:
        llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            azure_deployment=config.azure_chat_deployment,
            temperature=0,
        )
        return AzureLLMCategorizer(llm, fallback)
    return fallback
