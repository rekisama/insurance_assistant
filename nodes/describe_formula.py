from langchain.schema import AIMessage
from state import State
from config import pdf_path
import re

def describe_formula_node(state: State):
    """
    输出公式的自然语言描述
    - 遍历 state["formulas"]，用大模型将每条公式转为通俗易懂的自然语言说明
    - 回复用户
    """
    llm = state.get("llm")
    docs = state.get("docs")
    vectordb = state.get("vectordb")

    formulas = state.get("formulas", [])
    if not formulas:
        reply = AIMessage(content="当前未提取到任何保险合同公式。")
        return {"messages": state["messages"] + [reply]}
    desc_list = []
    for idx, f in enumerate(formulas, 1):
        formula = f.get("formula", "")
        variables = f.get("variables", {})
        # 构造变量说明字符串
        vars_expl = "，".join([f"{k}：{v}" for k, v in variables.items()])
        # 让LLM生成自然语言描述
        prompt = (
            f"请用通俗易懂的自然语言解释以下保险合同公式的含义和计算逻辑：\n"
            f"公式：{formula}\n"
            f"变量说明：{vars_expl}"
        )
        response = llm.invoke(prompt)
        desc = response.content.strip()
        desc_list.append(f"{idx}. 公式：{formula}\n   说明：{desc}")
    reply_text = "以下是保险合同中提取的公式及其自然语言说明：\n" + "\n\n".join(desc_list)
    reply = AIMessage(content=reply_text)
    return {"messages": state["messages"] + [reply]}