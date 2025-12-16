import os
import traceback

from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode

load_dotenv()
import json
from typing import Annotated
from langchain_community.chat_models.tongyi import ChatTongyi
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver

# 这是内存检查点，这对于本教程来说很方便
# 但是在生产环境中，需要将其改为 sqliterServer PostgresServer 或者 redis server 数据库

memory = InMemorySaver()


class State(TypedDict):
    messages: Annotated[list, add_messages]


# img. 创建 StateGraph
graph_builder = StateGraph(State)

tavily_tool = TavilySearch(
    tavily_api_key=os.getenv("Tavily_KEY"),
    max_results=2)

tools = [tavily_tool]

llm = ChatTongyi(
    model="qwen-max",
    api_key=os.getenv("Ali_KEY"),
    temperature=0
)

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# 添加一个 chatbot 节点
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode([tavily_tool])
graph_builder.add_node("tools", tool_node)


def route_tools(state: State):
    """
     Use in conditional_edge to route to the ToolNode if the last message
     has tool calls. Otherwise,rotue to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise TypeError(f"No messages found in input state to tool_edge:{state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {
        "tools": "tools", END: END
    }
)
graph_builder.add_edge("tools", "chatbot")
# 添加一个 entry 点来告诉图表每次运行时从哪里开始工作
graph_builder.add_edge(START, "chatbot")
# 添加一个 exit 点来只是图表应该在哪里结束
# graph_builder.add_edge("chatbot",END)

config = {
    "configurable": {
        "thread_id": "img"
    }
}

# 编译图
graph = graph_builder.compile(checkpointer=memory)

# 可视化图（可选）
from IPython.display import display, Image

try:
    display(Image(graph.get_graph().draw_png()))
except Exception as e:
    print(e)
    pass

# 运行聊天机器人
def steam_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config):
        for value in event.values():
            print("Assisant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User:")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye")
            break
        steam_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User:" + user_input)
        steam_graph_updates(user_input)
        break
