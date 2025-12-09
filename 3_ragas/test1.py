import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAI, ChatOpenAI
from langchain_classic.text_splitter import  RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from langchain_classic.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.schema.runnable import RunnableMap
from langchain_classic.schema.output_parser import StrOutputParser
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy

# 初始化对话大模型
chatLLM = ChatOpenAI(
    api_key="sk-66bc27a6330f434f8751f8172a73064f",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="deepseek-v3"
)

# 调用阿里百炼平台文本嵌入式模型，配置环境变量
embeddings = DashScopeEmbeddings(
    dashscope_api_key="sk-66bc27a6330f434f8751f8172a73064f",
    model="text-embedding-v1",
)
# 创建主文档分割器
parent_spliter = RecursiveCharacterTextSplitter(chunk_size=512)
# 创建子文档分割器
child_spliter = RecursiveCharacterTextSplitter(chunk_size=256)
# 创建向量数据库对象
vectorstore = Chroma(
    persist_directory="./chromadb",
    collection_name="split_parents",
    embedding_function=embeddings,
)
# 创建内存储存对象
store = InMemoryStore()
# 创建父文档检索器
reteriver = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_spliter,
    parent_splitter=parent_spliter,
    search_kwargs={"k":2}
)

docs = PyPDFLoader('LangGraph.pdf').load()
# 添加文档集
reteriver.add_documents(docs)
print(list(store.yield_keys()))


# 创建 prompt 模板
template = """
你是一个问答任务助手。利用以下检索到的上下文信息回答问题。
若不知道答案，只需回复 “不知道”。
回答最多两句话，保持简洁。
问题：{question}
上下文：{context}
回答：
"""
# 由模板生成 prompt

prompt = ChatPromptTemplate.from_template(template)
# 创建 chain (LCEL langchain 表达语言)
chain = RunnableMap({
     "context": lambda x:reteriver.invoke(x["question"]),
     "question": lambda x:x["question"]
 }) | prompt | chatLLM | StrOutputParser()

query = "langGraph是什么"
response = chain.invoke({"question": query})
print(response)

# 用户问的问题
questions=["LangGraph 的核心优势有哪些？",
          "LangGraph 的核心组件包括什么？",
          "LangGraph 平台的组成部分有哪些？",
          "如何在 LangGraph 中添加普通边？",
          "LangGraph 中 State（状态）的模式可以是什么类型？",
          "LangGraph 的灵感来源于哪些技术？"]

# 真实答案
ground_truths=["LangGraph 的核心优势包括循环性、可控性和持久性。",
        "LangGraph 的核心组件有 Graph（图）、State（状态）、Nodes（节点）和 Edges（边）。",
        "LangGraph 平台的组成部分有 LangGraph 服务器（API）、LangGraph SDK（API 客户端）、"
        "LangGraph CLI（构建服务器的命令行工具）、LangGraph Studio（用户界面 / 调试器）。",
        "可以使用 add_edge 方法添加普通边，例如 graph.add_edge (node_a,node_b)，表示总是从节点 A 到节点 B。",
        "State（状态）的模式可以是 TypedDict 或者 Pydantic 模型。",
        "LangGraph 的灵感来源于 Pregel 和 Apache Beam，其公共接口受 NetworkX 的启发。"]

answers = []
contexts = []

# Inference 推理
for query in questions:
    answers.append(chain.invoke({"question": query}))
    contexts.append([docs.page_content for docs in reteriver.invoke(query) ])

# To dict
data = {
    "user_input": questions,
    "response": answers,
    "retrieved_contexts": contexts,
    "reference": ground_truths,
}
# convert dict to dataset
dataset = Dataset.from_dict(data)
print(dataset)

result = evaluate(
    dataset=dataset,
    # 定义评估指标
    metrics=[
        context_precision,# 上下文精度
        context_recall,# 上下文召回率
        faithfulness,# 忠实度
        answer_relevancy,# 答案相关性
    ],
    llm=chatLLM,
    embeddings=embeddings,
)

df = result.to_pandas()
print(df)





