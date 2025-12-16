import os
from dotenv import load_dotenv

load_dotenv()
import json
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

import inspect

def get_weather(city: str) -> str:
    """Get weather from city"""
    return f"它是晴天 {city}!"


checkpoint = InMemorySaver()

model = ChatTongyi(
    model="qwen-max",
    api_key=os.getenv("Ali_KEY"),
    temperature=0
)

agent = create_agent(
    model=model,
    tools=[get_weather],
    checkpointer=checkpoint,
    system_prompt="你是个智能助手"
)

# run agent
config = {
    "configurable": {
        "thread_id": "img"
    }
}


print("Agent输入格式Schema：")
print(agent.input_schema.model_json_schema())


shanghai = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "上海的天气怎么样"}
        ]
    },
    config
)
print(shanghai)

hangzhou = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "杭州怎么样"}
        ]
    },
    config
)

print(hangzhou)
