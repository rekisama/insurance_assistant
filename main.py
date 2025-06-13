from config import pdf_path, marked_pdf_path, llm, docs, vectordb
from pdf_utils import *
from nodes import *
from langchain_core.messages import HumanMessage, AIMessage
from graph_builder import build_graph



def insurance_pdf_summary_chat():
    print("保险助手已启动，多轮对话模式，输入“退出”可结束。")
    # 初始化公式
    init_state = {
        "messages": [],
        "formulas": [],
        "user_inputs": {},
        "llm": llm,
        "pdf_path": pdf_path,
        "marked_pdf_path": marked_pdf_path,
        "docs": docs,
        "vectordb": vectordb,
    }
    formula_state = extract_formula_node(init_state)
    # 合并公式抽取到初始state
    state = {**init_state, **formula_state}

    graph = build_graph()

    # 输出初始化结果，提示用户
    if state.get("formulas"):
        print("（已自动提取保险合同中的计算公式，后续可直接进行参数补全和计算）")
    else:
        print("（未能自动提取保险合同中的计算公式）")

    while True:
        try:
            query = input("用户：")
            if query.strip().lower() in ["退出", "exit", "quit", "再见"]:
                print("保险助手：感谢使用，再见！")
                break
            msg = HumanMessage(content=query)
            state["messages"] = state.get("messages", []) + [msg]
            final_state = None
            for event in graph.stream(state, stream_mode="values"):
                final_state = event
            if final_state and "messages" in final_state:
                for msg in reversed(final_state["messages"]):
                    if isinstance(msg, AIMessage):
                        print("保险助手：", msg.content)
                        break
                state = final_state
        except Exception as e:
            print(f"保险助手：发生错误 - {e}")

if __name__ == "__main__":
    insurance_pdf_summary_chat()