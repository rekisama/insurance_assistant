from langchain.schema import AIMessage
from state import State

def chat_node(state: State):
    """
    通用对话节点
    作用：
    - 不做检索、不做业务逻辑处理，直接将用户输入交给LLM，返回AI自由问答回复。
    输入：
    - state["messages"]: 消息历史
    输出：
    - state["messages"]: 增加一条AI回复
    适用场景：
    - 用户提问未命中任何业务关键词时，兜底用自然对话
    """
    llm = state.get("llm")
    docs = state.get("docs")
    vectordb = state.get("vectordb")

    user_msg = state["messages"][-1].content if state["messages"] else ""
    response = llm.invoke(user_msg)
    reply = AIMessage(content=response.content)
    return {"messages": state["messages"] + [reply]}