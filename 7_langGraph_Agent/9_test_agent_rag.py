import os
from dotenv import load_dotenv
load_dotenv()
import traceback
from langchain_classic.chains import llm
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
import json
from typing import Annotated
from langchain_community.chat_models.tongyi import ChatTongyi
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_community.embeddings import DashScopeEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
import uuid
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate,HumanMessagePromptTemplate

# 1.加载语料
loader = PyMuPDFLoader('./LangGraph.pdf')
pages = loader.load_and_split()

# 2.切片
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True
)
texts = text_splitter.create_documents(
    [page.page_content for page in pages],
)

#  4.灌库
embeddings = DashScopeEmbeddings(
    dashscope_api_key=os.getenv("Ali_KEY"),
    model="text-embedding-v1",
)
db = FAISS.from_documents(texts, embeddings)
# 5.检索 top-5
retriever = db.as_retriever(search_kwargs={"k": 5})

# 6.创建检索工具
from langchain_core.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retriever_books_tools","从本地文档中查找LangGraph相关资料"
)

# 测试工具
# retriever_tool.invoke({
#     "query":"什么是 LangGraph?"
# })


# 生成查询
llm = ChatTongyi(
    model="qwen-max",
    api_key=os.getenv("Ali_KEY"),
    temperature=0
)

def generate_query_or_response(state:MessagesState):
    response = (
        llm.bind_tools([retriever_tool]).invoke(state["messages"])
    )
    return {"messages": [response]}

# 随机输入
# input = {"messages":[{"role":"user","content":"hello!"}]}
# generate_query_or_response(input)["messages"][-1].pretty_print()

# 提出一个需要语义搜索的问题：

input = {"messages":[{"role":"user","content":"什么是 langGraph 的条件边？"}]}
generate_query_or_response(input)["messages"][-1].pretty_print()

# 对检索结果文档进行评分
from pydantic import  BaseModel,Field
from typing import Literal

# 推荐使用""" 方式进行构造 prompt 模板

# grade_prompt = (
# "你是一个评审员，负责评估检索到的文档与用户问题的相关性。\n"
# "以下是检索到的文档:\n\n"
# "{context} \n\n"
# "以下是用户问题:{question}\n"
# "如果文档包含与用户问题相关的关键字或语义含义，将其评为相关。\n"
# "给出一个二元评分 'yes' 或 'no'，以指示文档是否与问题相关。"
# )

grade_prompt = """
你是一个评审员，负责评估检索到的文档与用户问题的相关性。
以下是检索到的文档:
{context}
以下是用户问题:{question}
如果文档包含与用户问题相关的关键字或语义含义，将其评为相关。
给出一个二元评分 'yes' 或 'no'，以指示文档是否与问题相关。
"""

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance checking."""
    binary_score :str = Field(description="Relevance score : 'yes' if relevant,or 'no' if not relevant")


def grade_documents(state:MessagesState) -> Literal["generate_answer","rewriter_question"]:
    """Determine whether the reterieved document is relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = grade_prompt.format(question=question,context=context)
    response = (
        llm.with_structured_output(GradeDocuments).invoke([HumanMessage(content=prompt)])
    )
    score = response.binary_score
    if score == "yes":
        return "generate_answer"
    else:
        return "rewriter_question"


 # 测试  grade_documents
from langchain_core.messages import convert_to_messages
input = {
    "messages": convert_to_messages(
        [{
            "role":"user",
            "content":"什么是 LangGraph"
        },
        {
            "role":"assistant",
            "content":"",
            "tool_calls":[
                {
                    "id":"1",
                    "name":"retriever_books_tools",
                    "args":{
                        "query":"什么是 LangGraph"
                    }
                }
            ]
        },
        {
            "role":"tool","content":"LangGraph","tool_call_id":"1"
        }]
    )
}
# print(grade_documents(input))

# 问题重写
rewrite_prompt = """
查看输入内容并尝试推理其潜在的语义意图/含义。
这是初始问题：
-------
{question}
-------
提出一个改进后的问题：
"""
def rewrite_question(state:MessagesState):
    """重写用户原始的问题"""
    messages= state["messages"]
    question = messages[0].content
    prompt = rewrite_prompt.format(question=question)
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [{"role":"user","content":response.content}]}


input = {
    "messages": convert_to_messages(
        [{
            "role":"user",
            "content":"什么是 LangGraph"
        },
        {
            "role":"assistant",
            "content":"",
            "tool_calls":[
                {
                    "id":"1",
                    "name":"retriever_books_tools",
                    "args":{
                        "query":"什么是 LangGraph"
                    }
                }
            ]
        },
        {
            "role":"tool","content":"newsf","tool_call_id":"1"
        }]
    )
}
# print(rewrite_question(input))

# 生成答案
generate_prompt = """
您是一个问答任务助理。使用以下检索到的上下文片段来回答问题。如果您不知道答案，请直接说不知道。最多使用三句话，并保持回答简洁。
问题: {question}
上下文: {context}
"""

def generate_answer(state:MessagesState):
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = generate_prompt.format(question=question,context=context)
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

input = {
    "messages": convert_to_messages(
        [{
            "role":"user",
            "content":"什么是 LangGraph"
        },
        {
            "role":"assistant",
            "content":"",
            "tool_calls":[
                {
                    "id":"1",
                    "name":"retriever_books_tools",
                    "args":{
                        "query":"什么是 LangGraph"
                    }
                }
            ]
        },
        {
            "role":"tool","content":"LangGraph 是一个用于构建有状态的多参与者应用程序的库，利用 LLM 创建代理和多代理工作流.",
            "tool_call_id":"1"
        }]
    )
}
# print(generate_answer(input))

# 构建工作流
workflow = StateGraph(MessagesState)
# 添加节点 node
workflow.add_node("generate_query_or_response",generate_query_or_response)
workflow.add_node("retrieve",ToolNode([retriever_tool]))
workflow.add_node("rewriter_question",rewrite_question)
workflow.add_node("generate_answer",generate_answer)
# 添加边
workflow.add_edge(START,"generate_query_or_response")
workflow.add_conditional_edges(
    "generate_query_or_response",
    tools_condition,
    {
        "tools":"retrieve",
        END:END
    }
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)
workflow.add_edge("generate_answer",END)
workflow.add_edge("rewriter_question","generate_query_or_response")

graph = workflow.compile()

#可视化图（可选）
from IPython.display import display,Image

try:
    display(Image(graph.get_graph().draw_png()))
except Exception as e:
    pass

# 运行 rag
for chunk in graph.stream(
        {
            "messages":[
                {
                    "role": "user",
                    # "content": "什么是 langGraph?"
                    "content": "什么是条件边?"
                }
            ]
        }
):
    for node,update in chunk.items():
        print("upadte from node",node)
        update["messages"][-1].pretty_print()
        print("\n\n")


