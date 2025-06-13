from typing_extensions import TypedDict
from typing import Annotated, List, Dict, Any
from langgraph.graph.message import add_messages
from langchain_deepseek import ChatDeepSeek

class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    formulas: List[Dict[str, Any]]
    user_inputs: Dict[str, Any]
    llm: Any
    pdf_path: str
    marked_pdf_path: str
    docs: Any
    vectordb: Any

llm = ChatDeepSeek(model="deepseek-chat")
