from langchain.schema import AIMessage
from state import State
from pdf_utils import extract_keywords, highlight

# --------- 混合检索节点 ---------
def vector_search_node(state: State):
    """
    混合检索节点
    作用：
    - 接收用户问题，先用向量数据库做语义检索（查找与问题最相关的片段），再用关键词遍历所有分段（兜底查找包含关键词的片段）。
    - 两路结果合并去重，输出最多3条最相关片段（高亮关键词并显示页码）。
    - 若未命中，提示未找到，并可据此判断是内容确无、分段有误还是embedding问题。
    输入：
    - state["messages"]: 消息历史，取最新一条为用户问题
    输出：
    - state["messages"]: 增加一条AI回复
    适用场景：
    - 保险合同内容智能检索、关键词兜底、FAQ等
    """
    llm = state.get("llm")
    docs = state.get("docs")
    vectordb = state.get("vectordb")

    user_msg = state["messages"][-1].content if state["messages"] else ""
    keywords = extract_keywords(user_msg)
    # 1. 语义向量检索
    vector_results = vectordb.similarity_search(user_msg, k=8)
    # 2. 关键词精确检索
    keyword_results = [doc for doc in docs if any(kw in doc.page_content for kw in keywords)]
    # 3. 合并去重，保持顺序
    doc_ids = set()
    all_results = []
    for doc in vector_results + keyword_results:
        doc_id = (doc.metadata.get("page", None), doc.page_content)
        if doc_id not in doc_ids:
            doc_ids.add(doc_id)
            all_results.append(doc)
    # 4. 输出
    if all_results:
        context = "\n\n".join([
            f"【第{doc.metadata.get('page', '?')}页】\n{highlight(doc.page_content, keywords)}"
            for doc in all_results[:3]
        ])
        prompt = (
            f"以下是与用户问题最相关的保险合同片段（含页码，已高亮关键词）：\n{context}\n\n"
            f"请结合这些内容（引用原文时请带页码），回答用户问题：“{user_msg}”"
        )
        response = llm.invoke(prompt)
        reply = AIMessage(content=response.content)
    else:
        reply = AIMessage(content="未找到相关保险条款片段（已全文检索）。")
    return {"messages": state["messages"] + [reply]}