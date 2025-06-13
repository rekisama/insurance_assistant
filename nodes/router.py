from state import State

def router_node(state: State):
    return state

def route_fn(state: State):
    msg = state["messages"][-1].content
    llm = state.get("llm")  # 推荐从state取llm，而不是全局

    if any(kw in msg for kw in visualize_keywords):
        return "llm_insurance_visualize"
    if any(kw in msg for kw in risk_keywords):
        return "pdf_risk_mark"
    elif any(kw in msg for kw in summary_keywords):
        return "pdf_summary"
    elif any(kw in msg for kw in vector_keywords):
        return "vector_search"
    elif any(kw in msg for kw in describe_formula_keywords):
        return "describe_formula"

    intent_prompt = (
        f"你是保险智能助手。用户输入：{msg}\n"
        "请判断用户是否希望看到保险利益的可视化（如图表、走势、变化曲线等），如果是请回复YES，否则回复NO："
    )
    if llm and "YES" in llm.invoke(intent_prompt).content.upper():
        return "llm_insurance_visualize"

    intent_prompt = (
        f"你是保险智能助手。用户输入：{msg}\n"
        "请判断这个问题是否需要你综合保险合同内容和用户个人情况自由推理解答，如果需要请回复YES，否则回复NO："
    )
    if llm and "YES" in llm.invoke(intent_prompt).content.upper():
        return "llm_free_reasoning"
    return "chat"

summary_keywords = ["总结", "摘要", "概括", "主要内容", "要点", "保险合同内容", "条款", "保障范围"]
vector_keywords = ["查找", "查阅", "具体内容", "详细内容", "关键词", "关键字", "找到", "规定", "责任", "义务", "赔偿", "免赔"]
risk_keywords = ["高亮", "标记", "标注", "风险", "免责", "理赔限制", "重点内容", "生成标注PDF"]
describe_formula_keywords = ["解释公式", "描述公式", "公式说明", "公式含义", "公式描述", "公式计算"]
visualize_keywords = ["可视化", "画图", "图表", "走势", "趋势", "变化曲线", "benefit visualization", "plot", "visualize"]
