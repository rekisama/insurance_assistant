from pdf_utils import load_and_split_pdf, build_vectorstore
from langchain_deepseek import ChatDeepSeek

from dotenv import load_dotenv
load_dotenv()

pdf_path = "example.pdf"
marked_pdf_path = "example_marked.pdf"
llm = ChatDeepSeek(model="deepseek-chat")

docs = load_and_split_pdf(pdf_path)
vectordb = build_vectorstore(docs)