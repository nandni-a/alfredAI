# intent_utils.py

from langchain_core.messages import HumanMessage
from base_reasoning import get_llm

llm = get_llm("openai")

def detect_intent(message: str) -> str:
    """
    Uses LLM to classify user intent into one of the categories:
    - 'route' : for directions or map navigation
    - 'nearby' : for nearby places search
    - 'search' : for general web lookup
    - 'chat' : default LLM chat
    """
    system_prompt = (
        "You are an intent detection system."
        "Classify the user's input strictly as one of: 'route', 'nearby', 'search', or 'chat'."
        "Respond only with the label."
    )
    prompt = f"User: {message}\nIntent:"
    response = llm.invoke([HumanMessage(content=system_prompt), HumanMessage(content=prompt)])
    return response.content.strip().lower()
