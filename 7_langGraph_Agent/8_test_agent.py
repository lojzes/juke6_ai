import os
import traceback

from dotenv import load_dotenv
from langchain_classic.chains import llm
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

load_dotenv()
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

loader = PyMuPDFLoader('./LangGraph.pdf')
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True
)
texts = text_splitter.create_documents(
    [page.page_content for page in pages],
)

llm = ChatTongyi(
    model="qwen-max",
    api_key=os.getenv("Ali_KEY"),
    temperature=0
)

# 灌库
embeddings = DashScopeEmbeddings(
    dashscope_api_key=os.getenv("Ali_KEY"),
    model="text-embedding-v1",
)
db = FAISS.from_documents(texts, embeddings)
# 检索 top-5
retriever = db.as_retriever(search_kwargs={"k": 5})

from langchain_core.prompts import ChatPromptTemplate,HumanMessagePromptTemplate

template = """
请根据对话历史和下面提供的信息回答上面用户提出的问题:
{query}
"""
prompt = ChatPromptTemplate(
    [
        HumanMessagePromptTemplate.from_template(template),
    ]
)

def retrieval(state:MessagesState):
    user_query = ""
    if len(state["messages"]) >= 1:
        # 获取最后一轮用户输入
        user_query = state["messages"][-1]
    else:
        return {"messages": []}
    #检索
    docs = retriever.invoke(str(user_query))
    #填充 prompt 模板
    messages = prompt.invoke("\n".join([doc.page_content for doc in docs])).messages
    return {"messages": messages}


def chatbot(state:MessagesState):
    return {"messages":[llm.invoke(state["messages"])]}


# graph_builider = StateGraph(MessagesState)
# graph_builider.add_node("retriever",retriever)
# graph_builider.add_node("chotbot",chatbot)
#
# graph_builider.add_edge(START,"retriever")
# graph_builider.add_edge("retriever","chotbot")
# graph_builider.add_edge("chotbot",END)
#
# graph = graph_builider.compile()

# 可视化图（可选）
# from IPython.display import display,Image
#
# try:
#     display(Image(graph.get_graph().draw_png()))
# except Exception as e:
#     pass


from langchain_classic.schema import HumanMessage
from typing import Literal
from langgraph.types import interrupt,Command

# 校验
def verify(state:MessagesState)->Literal["chatbot","ask_human"]:
    messages = HumanMessage("请根据对话历史和上面提供的信息判断，已知的的信息是否能回答用户的问题，直接输出你的判断'Y'或者'N'")
    ret = llm.invoke(state["messages"]+[messages])
    if 'Y' in ret.content:
        return "chatbot"
    else:
        return "ask_human"


# 人工处理
def ask_human(state:MessagesState):
    user_query = state["messages"][-2].content
    human_response = interrupt(
        {
            "question":user_query,
        }
    )
    # Update the state with the human input or route the graph based on the input
    return {
        "messages":[AIMessage(human_response)],
    }


def chatbot(state:MessagesState):
    return {"messages":[llm.invoke(state["messages"])]}


memory = MemorySaver()

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("retrieval",retrieval)
graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("ask_human",ask_human)

graph_builder.add_edge(START,"retrieval")
graph_builder.add_conditional_edges("retrieval",verify)
graph_builder.add_edge("ask_human",END)
graph_builder.add_edge("chatbot",END)

# 中途会被转人工，所以需要 checkpointer 存储状态
graph = graph_builder.compile(checkpointer=memory)

# 可视化图（可选）
from IPython.display import display,Image

try:
    display(Image(graph.get_graph().draw_png()))
except Exception as e:
    pass

thead_config = {
    "configurable":{
        "thread_id":"1"
    }
}

def stream_graph_updates(user_input:str):
    # 向 graph 传入一条消息（触发状态跟新 add_messages）
    for event in graph.stream(
            {"messages":[
                {"role":"user","content":user_input},
            ]},
            thead_config
    ):
        for value in event.values():
            if isinstance(value, tuple):
                return value[0].value["question"]
            elif "messages" in value and isinstance(value["messages"][-1], AIMessage):
                print("Assistant:",value["messages"][-1].content)
    return None

def resume_graph_updates(human_input:str):
    for event in graph.stream(
        Command(resume=human_input),thead_config,stream_mode="updates"
    ):
        for value in event.values():
            if "messages" in value and isinstance(value["messages"][-1], AIMessage):
                print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User:")
        if user_input == "exit":
            break
        question = stream_graph_updates(user_input)
        if question:
            human_answer = input("Ask Human:" + question + "\nHuman: ")
            resume_graph_updates(human_answer)
    except Exception as e:
        traceback.print_exc()