# 保险合同智能助手（Insurance Contract Assistant）

本项目基于大语言模型（DeepSeek LLM）与LangGraph多节点流程编排，实现了对保险合同 PDF 的智能分析与问答。支持合同内容摘要、语义和关键词混合检索、风险条款高亮、公式结构化提取、保险利益可视化等多种功能。
个人练手项目，用于学习交流，欢迎参考学习。

## 功能特色

- **合同自动摘要**：对保险合同 PDF 主要内容和保障范围进行要点提取。
- **智能检索问答**：基于向量库（FAISS + HuggingFaceEmbeddings）和关键词兜底，支持语义和关键词混合检索，智能回答用户关于合同内容的问题。
- **风险条款高亮**：自动识别高风险、免责等重要条款，并生成高亮标注后的 PDF 文件。
- **公式结构化与说明**：自动抽取合同中的金钱计算/赔付/现金价值等公式并结构化，结合自然语言解释公式含义。
- **保险利益可视化**：根据历史对话，自动生成年度保险利益表格及收益曲线图。
- **多轮对话与自由推理**：支持上下文多轮对话，LLM 可对复杂或开放式问题进行智能推理。

## 目录结构

```
insurance_assistant/
├── main.py                 # 入口程序
├── config.py               # 配置与全局变量
├── pdf_utils.py            # PDF 处理工具
├── graph_builder.py        # 流程图构建
├── nodes/                  # 各功能节点
├── requirements.txt        # 依赖列表
├── .env                    # 环境变量
├── .gitignore
└── README.md
```

## 环境依赖

- Python 3.8+
- langchain
- langgraph
- faiss-cpu
- python-dotenv
- jieba
- PyMuPDF
- huggingface_hub
- [DeepSeek LLM API 权限](https://platform.deepseek.com/)
- 其他详见 `requirements.txt`

安装依赖：
```bash
pip install -r requirements.txt
```

## 快速开始

1. **准备 .env 文件**

项目根目录下创建 `.env` 文件，内容形如：

```
DEEPSEEK_API_KEY=你的_api_key
```

2. **准备 PDF 合同文件**

将需要分析的保险合同 PDF 放在根目录路径（如 `example.pdf`）。

3. **运行主程序**

```bash
python main.py
```

交互示例：
```
保险助手已启动，多轮对话模式，输入“退出”可结束。
用户：请帮我总结一下这份保险合同的主要条款
保险助手：（输出合同摘要）
用户：查找本合同的免责条款
保险助手：（输出相关条款及页码、关键词高亮）
```

## 注意事项

- `.env` 文件包含敏感密钥，请勿上传至 GitHub。
- 本项目仅供学习与技术交流，实际保险条款解释请以保险公司官方文件为准。

## 参考资料

- [DeepSeek LLM 官方文档](https://platform.deepseek.com/)
- [LangChain 官方文档](https://python.langchain.com/)
- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [FAISS 官方文档](https://faiss.ai/)
- [PyMuPDF 文档](https://pymupdf.readthedocs.io/)

---

如有建议或问题，欢迎 issue 交流！
