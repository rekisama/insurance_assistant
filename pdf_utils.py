from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

# --------- 加载并分段 ---------
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()
    for idx, doc in enumerate(docs):
        if "page" not in doc.metadata:
            doc.metadata["page"] = idx + 1
    return docs

# --------- 向量库初始化 ---------
def build_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectordb = FAISS.from_documents(docs, embedding=embeddings)
    return vectordb


# --------- 关键词提取 ---------
def extract_keywords(user_msg):
    try:
        import jieba.analyse
        return jieba.analyse.extract_tags(user_msg, topK=8)
    except Exception:
        # 简单按空格/标点切分
        return [w for w in re.split(r"[，。,.？?、\s]", user_msg) if w.strip()]

# --------- 高亮函数 ---------
def highlight(text, keywords):
    for kw in keywords:
        if kw.strip():
            text = re.sub(
                re.escape(kw),
                f"**{kw}**",
                text,
                flags=re.IGNORECASE
            )
    return text