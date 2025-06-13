from langchain.schema import AIMessage
from state import State
from typing import Optional

def llm_free_reasoning_node(state):
    """
    LLM自由推理节点
    - 综合全部历史对话和合同全文，直接让LLM给出判断、解答或建议，不依赖结构化公式。
    """
    llm = state.get("llm")
    docs = state.get("docs")
    vectordb = state.get("vectordb")
    
    # 拼接全部历史对话
    if state.get("messages"):
        dialog_history = ""
        for m in state["messages"]:
            # 如有角色，带上角色标注，更利于LLM理解
            if hasattr(m, "role"):
                dialog_history += f"{m.role}: {m.content}\n"
            else:
                dialog_history += f"{m.content}\n"
    else:
        dialog_history = ""
    pdf_text = "\n".join([doc.page_content for doc in docs])
    llm = state.get("llm")
    if not llm or not dialog_history.strip():
        reply = AIMessage(content="缺少llm对象或对话内容，无法进行智能推理。")
        return {"messages": state.get("messages", []) + [reply]}

    prompt = (
        "你是一位保险合同智能顾问。\n"
        "以下为保险合同的完整内容和全部历史对话，请结合用户的个人情况和问题，"
        "直接基于合同内容为用户给出专业判断、建议或答复，无需输出具体公式。\n\n"
        f"保险合同内容：\n{pdf_text}\n\n"
        f"历史对话：\n{dialog_history}\n"
        "请以专业、通俗、准确的语言回答用户。"
    )
    response = llm.invoke(prompt)
    reply = AIMessage(content=response.content)
    return {"messages": state["messages"] + [reply]}