import os
import traceback

from dotenv import load_dotenv
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
from langgraph.store.memory import  InMemoryStore
from langchain_community.embeddings import DashScopeEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
import uuid
from langchain_core.runnables import RunnableConfig

# 这是内存检查点，这对于本教程来说很方便
# 但是在生产环境中，需要将其改为 sqliterServer PostgresServer 或者 redis server 数据库

memory = InMemorySaver()
in_memory_store = InMemoryStore(
    index={
        "embed": DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=os.getenv("Ali_KEY"),
        ),
        "dims":1536
    }
)

llm = ChatTongyi(
    model="qwen-max",
    api_key=os.getenv("Ali_KEY"),
    temperature=0
)

def call_model(state:MessagesState,config:RunnableConfig,*,store:BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("memories",user_id)
    memories = store.search(namespace,query=str(state["messages"][-1].content))
    info = "\n".join([d.value["data"] for d in memories])
    system_msg = f"You are a helpfull assisant talking to the user.User info: {info}"
    # Store new memories if the user asks the model to remember
    last_message = state["messages"][-1]
    if "remember" in last_message.content.lower():
        memory = "User name is lojzes"
        store.put(namespace,str(uuid.uuid4()),{"data":memory})

    response = llm.invoke(
        [{"role":"system","content":system_msg}] + state["messages"],
    )
    return {"messages":response}

builder = StateGraph(MessagesState)
builder.add_node("call_model",call_model)
builder.add_edge(START,"call_model")
graph = builder.compile(checkpointer=MemorySaver(),store=in_memory_store)

config = {"configurable":{"thread_id":"1","user_id":"1"},}
input_message = {"role":"user","content":"我是 lojzes"}
for chunk in graph.stream({"messages":[input_message]},config,stream_mode="values"):
    chunk["messages"][-1].pretty_print()


config = {"configurable":{"thread_id":"1","user_id":"1"},}
input_message = {"role":"user","content":"我是谁"}
for chunk in graph.stream({"messages":[input_message]},config,stream_mode="values"):
    chunk["messages"][-1].pretty_print()

