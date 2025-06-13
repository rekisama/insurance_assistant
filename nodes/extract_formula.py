from langchain.schema import AIMessage
from state import State
from config import pdf_path
import re

def extract_formula_node(state: State):
    """
    节点作用：
    - LLM通读整个保险合同文本，自动提取所有涉及金钱计算/赔付/现金价值/保费等规则；
    - 用Python公式表达，并输出变量名及含义，结构化存储，便于后续变量补全和自动计算。
    用法：
    - 用户说“我要计算”或“帮我提取合同公式”时触发
    - state["formulas"] 将变为 [{formula, variables}] 结构
    """
    # 合并所有PDF文本
    llm = state.get("llm")
    docs = state.get("docs")
    vectordb = state.get("vectordb")
    pdf_text = "\n".join([doc.page_content for doc in docs])
    prompt = (
        "请从以下保险合同内容中，提取所有涉及金钱计算、赔付、现金价值、保费、给付等规则，"
        "并用Python公式表示。每条公式请用如下格式输出：\n"
        "【公式】：insurance_payment = base_amount * payout_ratio\n"
        "【变量】：base_amount=基本保额；payout_ratio=赔付比例\n"
        "如果有多条规则，请每条都这样写并换行。只输出公式和变量，不要输出解释。\n"
        "保险合同内容如下：\n" + pdf_text
    )
    response = llm.invoke(prompt)
    formulas = []
    # 解析LLM输出为结构化数据
    blocks = re.split(r"[【】]", response.content)
    for i in range(len(blocks)):
        if blocks[i] == "公式" and i+2 < len(blocks) and blocks[i+2] == "变量":
            formula = blocks[i+1].strip()
            varstr = blocks[i+3].strip()
            var_items = [v for v in varstr.split("；") if "=" in v]
            variables = {}
            for item in var_items:
                k, v = item.split("=", 1)
                variables[k.strip()] = v.strip()
            formulas.append({"formula": formula, "variables": variables})
    reply = AIMessage(content="已提取计算公式。")
    return {"messages": state["messages"] + [reply], "formulas": formulas}