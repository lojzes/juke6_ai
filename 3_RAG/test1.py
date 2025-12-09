import os
import logging
import pickle
from PyPDF2 import PdfReader
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI, ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.callbacks.manager import get_openai_callback
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing import List, Tuple


def extract_text_with_page_number(pdf) -> Tuple[str, List[int]]:
    """
    从 pdf 中提取文本并记录每行对应的页码
    参数：
      pdf: Pdf 文件对象
    返回：
      text: 提取的文本内容
      page_numbers:每行文本对应的页码列表

    """
    text = ""
    page_numbers = []

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
            page_numbers.extend([page_number] * len(extracted_text.split("\n")))

        else:
            logging.warning(f"No text found on page {page_number}.")

    return text, page_numbers


def process_text_with_splitter(text: str, page_numbers: List[int], save_path: str = None) -> FAISS:
    """
    处理文本并创建向量存储
    参数：
      text: 提取的文本
      page_numbers: 每行文本对应的页码
      save_path: 可选，保存向量数据库的路径
    返回：
      knowledgeBase: 基于 FAISS 的向量存储对象

    """
    text_spiltter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=512,
        chunk_overlap=128,
        length_function=len,

    )

    # 分割文本
    chunks = text_spiltter.split_text(text)
    logging.debug(f"Text split into {len(chunks)} chunks")
    print(f"文本被分成 {len(chunks)} 个块")

    # 创建嵌入式模型，OpenAI 嵌入式模型，配置环境变量 OPEN_AI_KEY
    # embeddings = OpenAIEmbeddings()
    # 调用阿里百炼平台文本嵌入式模型，配置环境变量
    embeddings = DashScopeEmbeddings(
        dashscope_api_key="sk-66bc27a6330f434f8751f8172a73064f",
        model="text-embedding-v1",
    )
    # 从文本块创建知识库
    knowledgeBase = FAISS.from_texts(chunks, embedding=embeddings)
    print("已从文本块创建知识库...")
    # 存储每个文本块对应的页码信息
    page_info = {chunk: page_numbers[i] for i, chunk in enumerate(chunks)}
    knowledgeBase.page_info = page_info
    # 如果提供了保存路径，则保存向量数据库和页码信息
    if save_path:
        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)
        # 保存FAISS向量数据库
        knowledgeBase.save_local(save_path)
        print(f"向量数据库已经保存到 {save_path}")
        # 保存页码信息到同一目录
        with open(os.path.join(save_path, "page_info.pkl"), "wb") as f:
            pickle.dump(page_info, f)
        print(f"页码信息已经保存到:{os.path.join(save_path, 'page_info.pkl')}")

        return knowledgeBase


def load_knowledge_base(load_path: str, embeddings=None) -> FAISS:
    """
    从磁盘加载向量数据库和页码信息

    参数：
      load_path：向量数据库的保存路径
      embeddings: 可选，嵌入模型。如果未None，将创建一个新的DashScopeEmbeddings实例

    返回：
      knowledgeBase: 加载的FAISS向量数据库对象
    """

    # 如果没有提供嵌入式模型,则创建一个新的
    if embeddings is None:
        embeddings = DashScopeEmbeddings(
            dashscope_api_key="sk-66bc27a6330f434f8751f8172a73064f",
            model="text-embedding-v1",
        )

    # 加载FAISS向量数据库，添加 allow_dangerous_deserialization=True参数以允许序列化
    knowledgeBase = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    print(f"向量数据库已从 {load_path} 中加载")
    # 加载页码信息
    page_info_path = os.path.join(load_path, "page_info.pkl")
    if os.path.exists(page_info_path):
        with open(page_info_path, "rb") as f:
            page_info = pickle.load(f)
        knowledgeBase.page_info = page_info
        print("页码信息已经加载")

    else:
        print("警告：未找到页码信息文件")

    return knowledgeBase


# 读取pdf 文件
pdf_reader = PdfReader("./LangGraph.pdf")
# 提取文本和页码信息
text, page_numbers = extract_text_with_page_number(pdf_reader)

print(text)


print(f"提取的文本长度：{len(text)} 个字符")

# 处理文本并创建知识库，同时保存到磁盘
save_dir = "./vector_db"
knowledgeBase = process_text_with_splitter(text,page_numbers,save_path=save_dir)


# 设置查询问题
query = "langgrap 是什么"

if query:
    # 执行相似度搜索，找到与查询相关的文档 top_n 、top_k 一个意思
    # 根据经验 k 一般5个左右
    docs = knowledgeBase.similarity_search(query,k=5)
    # 初始化对话大模型
    chatLLM = ChatOpenAI(
        api_key="sk-66bc27a6330f434f8751f8172a73064f",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="deepseek-v3"
    )
    # 加载问答链
    chain = load_qa_chain(chatLLM,chain_type="stuff")
    # 准备输入流
    input_data = {"input_documents":docs,"question":query}
    # 使用回调函数跟踪API调用成本
    with get_openai_callback() as cost:
        #执行问答链
        response = chain.invoke(input = input_data)
        print(f"查询已处理，成本：{cost}")
        print(response["output_text"])
        print("来源:")
    #记录唯一的页码
    unique_pages = set()
    #显示每个文档的来源页码
    for doc in docs:
        text_content = getattr(doc,"page_content","")
        source_page = knowledgeBase.page_info.get(
            text_content.strip(),"未知"
        )
        if source_page not in unique_pages:
            unique_pages.add(source_page)
            print(f"文本页码：{source_page}")

