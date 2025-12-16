import os
import traceback

from dotenv import load_dotenv

load_dotenv()
import json
from typing import Annotated
from langchain_community.chat_models.tongyi import ChatTongyi
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list,add_messages]

# img. 创建 StateGraph
graph_builder = StateGraph(State)

llm = ChatTongyi(
    model="qwen-max",
    api_key=os.getenv("Ali_KEY"),
    temperature=0
)

def chatbot(state:State):
    return {"messages":[llm.invoke(state["messages"])]}

# 添加一个 chatbot 节点
graph_builder.add_node("chatbot", chatbot)
# 添加一个 entry 点来告诉图表每次运行时从哪里开始工作
graph_builder.add_edge(START,"chatbot")
# 添加一个 exit 点来只是图表应该在哪里结束
graph_builder.add_edge("chatbot",END)
# 编译图
graph = graph_builder.compile()

# 可视化图（可选）
from IPython.display import display,Image

try:
    display(Image(graph.get_graph().draw_png()))
except Exception as e:
    pass

# 运行聊天机器人
def steam_graph_updates(user_input:str):
    for event in graph.stream({"messages":[{"role":"user","content":user_input}]}):
        for value in event.values():
            print("Assisant:",value["messages"][-1].content)


while True:
    try:
        user_input = input("User:")
        if user_input.lower() in ["quit","exit","q"]:
            print("Goodbye")
            break
        steam_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User:" + user_input)
        steam_graph_updates(user_input)
        break




