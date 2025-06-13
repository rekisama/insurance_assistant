from langchain.schema import AIMessage
from state import State


def pdf_summary_node(state: State):
    user_msg = state["messages"][-1].content if state["messages"] else ""
    docs = state.get("docs", [])
    llm = state.get("llm")
    pdf_text = "\n".join([doc.page_content for doc in docs])
    prompt = (
        f"请根据用户指令“{user_msg}”，对以下保险合同PDF内容做摘要。"
        "说明包含哪些主要条款和保障范围，重点条款需简要列出：\n"
        + pdf_text
    )
    response = llm.invoke(prompt)
    reply = AIMessage(content=response.content)
    return {"messages": state["messages"] + [reply]}