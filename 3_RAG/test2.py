import os
from dotenv import load_dotenv
load_dotenv()
import logging
import pickle
from PyPDF2 import PdfReader
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI, ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.callbacks.manager import get_openai_callback
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing import List, Tuple

# 初始化对话大模型
chatLLM = ChatOpenAI(
    api_key=os.getenv("Ali_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="deepseek-v3"
)

# 调用阿里百炼平台文本嵌入式模型，配置环境变量
embeddings = DashScopeEmbeddings(
    dashscope_api_key=os.getenv("Ali_KEY"),
    model="text-embedding-v1",
)
load_path = "./vector_db"
# 加载FAISS向量数据库，添加 allow_dangerous_deserialization=True参数以允许序列化
vectorstore = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
print(f"向量数据库已从 {load_path} 中加载")

# 创建 MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=chatLLM
)

query = "langgraph是什么"
# 执行查询
result = retriever.invoke(query)
# 打印结果
print(f"查询 {result}")
print(f"找到 {len(result)} 个相关文档")
for i,doc in enumerate(result):
    print(f"\n文档 {i + 1}")
    print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)