from langchain.schema import AIMessage
from state import State
from config import pdf_path
from typing import Optional
import re
import os

def llm_insurance_visualize_node(state):
    """
    Node：保险利益可视化
    - 结合 LLM，根据用户保险相关自然语言输入（支持全部历史对话），自动生成结构化利益表+matplotlib可视化代码并执行，输出图片路径。
    - state["messages"] 需包含对话历史，llm 对象需在 state["llm"]。
    - 可嵌入流程图/智能体链路中作为可视化节点。
    """
    llm = state.get("llm")
    docs = state.get("docs")
    vectordb = state.get("vectordb")

    # 拼接全部历史对话
    if state.get("messages"):
        dialog_history = ""
        for m in state["messages"]:
            # 若有"role"，则带上角色，提升上下文可读性
            if hasattr(m, "role"):
                dialog_history += f"{m.role}: {m.content}\n"
            else:
                dialog_history += f"{m.content}\n"
    else:
        dialog_history = ""

    llm = state.get("llm")
    output_img = state.get("output_img", "output.png")
    if not llm or not dialog_history.strip():
        reply = AIMessage(content="缺少llm对象或对话内容，无法生成可视化。")
        return {"messages": state.get("messages", []) + [reply]}

    prompt = f"""
请根据以下全部历史对话内容，理解用户的保险需求和合同描述，无论险种、利益类型、交费方式/领取方式如何变化，都请自动分析出每个保单年度的关键利益数据（如：年度、年龄、保费支出、养老金、祝寿金、现金价值、身故保险金等，若某项无则为0或空），输出结构化表格（如 pandas DataFrame），再用 matplotlib 画出主要利益随年份变化的折线图（如养老金、祝寿金、现金价值等），图片保存为 {output_img}。只输出完整可运行的 python 代码，不要解释。

历史对话如下：
{dialog_history}
"""
    llm_response = llm.invoke(prompt).content

    # 提取代码块
    code_match = re.search(r"```python(.*?)```", llm_response, re.DOTALL)
    if code_match:
        code = code_match.group(1)
    else:
        # 兜底：如果没有代码块就直接用全部回复
        code = llm_response

    # 执行代码
    exec_globals = {}
    try:
        exec(code, exec_globals)
    except Exception as e:
        reply = AIMessage(content=f"自动执行LLM生成代码出错：{e}\n代码如下：\n{code}")
        return {"messages": state.get("messages", []) + [reply]}

    # 检查图片文件
    if os.path.exists(output_img):
        reply = AIMessage(content=f"保险利益可视化已生成，图片文件：{output_img}")
        return {"messages": state.get("messages", []) + [reply], "output_img": output_img}
    else:
        reply = AIMessage(content="未找到生成的图片文件，可能LLM代码未正确执行。")
        return {"messages": state.get("messages", []) + [reply]}
    