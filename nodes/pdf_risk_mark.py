from langchain.schema import AIMessage  
from state import State
from config import pdf_path, marked_pdf_path
import fitz  # PyMuPDF

def pdf_risk_mark_node(state: State):
    """
    PDF风险条款高亮节点
    作用：
    - 将合同全文交给LLM，让其自动找出每页的高风险、免责、理赔限制等句子，并按页码输出。
    - 用PyMuPDF扫描每页原文并高亮LLM标记的关键句，生成高亮PDF文件。
    输入：
    - state["messages"]: 消息历史（不直接用），全局docs、pdf_path
    输出：
    - state["messages"]: 增加一条AI回复，提示已生成高亮PDF及存储路径
    适用场景：
    - 合同重点条款风险提示、合规检查
    """
    llm = state.get("llm")
    docs = state.get("docs")
    vectordb = state.get("vectordb")
    pdf_texts = [doc.page_content for doc in docs]
    prompt = (
        "你是保险合同分析专家。请找出下列PDF文本中的所有高风险条款、重要免责条款、理赔限制条款，"
        "以列表形式输出每页的高风险句子，格式如下：\n"
        "页码: 1\n风险内容: xxx; yyy\n页码: 2\n风险内容: zzz; ...\n"
        "PDF内容如下：\n" + "\n".join([f"第{i+1}页：{t}" for i, t in enumerate(pdf_texts)])
    )
    response = llm.invoke(prompt)
    mark_result = response.content

    page_risk_map = {}
    for block in mark_result.split("页码: "):
        block = block.strip()
        if not block: continue
        lines = block.split('\n')
        page_line = lines[0]
        risk_line = next((l for l in lines if l.startswith("风险内容:")), None)
        if risk_line:
            try:
                page_idx = int(page_line.strip()) - 1
            except Exception:
                continue
            risks = [s.strip() for s in risk_line.replace("风险内容:", "").split(";") if s.strip()]
            if risks:
                page_risk_map[page_idx] = risks

    doc = fitz.open(pdf_path)
    for page_idx, risks in page_risk_map.items():
        if page_idx < 0 or page_idx >= len(doc): continue
        page = doc[page_idx]
        text = page.get_text()
        for risk in risks:
            for inst in page.search_for(risk):
                page.add_highlight_annot(inst)
    doc.save(marked_pdf_path)

    reply = AIMessage(content=f"已为合同PDF高亮标注关键风险条款，保存为：{marked_pdf_path}")
    return {"messages": state["messages"] + [reply]}