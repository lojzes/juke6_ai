from langchain_core.tools import tool
from langchain_classic.chat_models import init_chat_model
import json

@tool
def add(a: int, b: int) -> int:
    """Add two integers.

      Args: a:First integer
            b:Second integer I
     """
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two integers.

        Args:
            a:First integer
            b:Second integer
    """

    return a * b


from langchain_core.messages import (
    AIMessage,  # 等价于 OpenAI 接口中的 assisant role
    HumanMessage,  # 等价于 OpenAI 接口中的 user role
    SystemMessage  # 等价于 OpenAI 接口中的 system role
)
llm = init_chat_model('deepseek-chat',model_provider='deepseek',
        api_key= "sk-fa1e532719fc46aa83166d15619bb76a"
    )

llm_with_tools = llm.bind_tools([add, multiply])
query = "3.5的 4 倍是多少？"
messages = [HumanMessage(query)]

output = llm_with_tools.invoke(messages)
print(json.dumps(output.tool_calls, indent=4))

messages.append(output)
available_tools = {"add": add, "multiply": multiply}

for tool_call in output.tool_calls:
    selected_tool = available_tools[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)

new_output = llm_with_tools.invoke(messages)
for message in messages:
    print(json.dumps(message.model_dump(), indent=4, ensure_ascii=False))

print(new_output.content)

