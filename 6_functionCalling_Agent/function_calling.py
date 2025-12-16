import os
from dotenv import load_dotenv
load_dotenv()

import json
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("Ali_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 工具定义
tools = [
    {
        #类型(固定格式)
        "type":"function",
        # 函数定义
        "function":{
            # 函数名称(帮助我们去查找本地的函数在哪里，函数映射的ID)
            "name":"get_weather",
            #函数描述(帮助模型理解函数的作用，适用场景，可以理解为Prompt的一部分)
            "description":"获取提供坐标的当前温度(摄氏度)",
            # 函数依赖的参数的定义(帮助模型理解如果要做参数生成，应该怎么生成)
            "parameters":{
                #参数形式
                "type":"object",# 对应输出JSON string
                # 参数结构
                "properties":{
                    #参数名称
                    "latitude":{"type":"number"},
                    "longlitude":{"type":"number"}
                },
                      #必须保证生成的参数列表
            "required":["latitude","longlitude"],
            "additionalProperties":False
            },
            # 格式是否严格
            "strict":True
        }
    }
]

messages = [{"role":"user","content":"今天上海的天气怎么样？"}]
#messages = [{"role":"user","content":"你今天怎么样？"}]
#messages = [{"role":"user","content":"请告诉我北京的经纬度"}]
# messages = [{"role":"user","content":"今天天气怎么样？"}]

completion = client.chat.completions.create(
    model="qwen3-coder-plus",
    messages=messages,
    tools=tools
)

if completion.choices[0].message.tool_calls:
    print(completion.choices[0].message.tool_calls[0].function)
else:
    print("没有函数调用")

function_calling_message = completion.choices[0].message
function_calling = completion.choices[0].message.tool_calls[0]
print("Call Function Name :",function_calling.function.name)
print("Call Function Arguments :",function_calling.function.arguments)

def get_weather(*,latitude:float,longlitude:float):
    return {
        "temperature":23,
        "weather":"晴天",
        "wind_direction":"东南风",
        "windy":2
    }
functions = {
    "get_weather":get_weather
}

function_res = functions[function_calling.function.name](**json.loads(function_calling.function.arguments))
# print(function_res)

# 必须：让模型知道自己之前给了一个什么指令（包含 tool_call_id）

messages.append(function_calling_message)
# 包含了tool_call_id 的结果加入消息队列
messages.append(
   {
        "role":"tool",
        "tool_call_id":function_calling.id,
        "content":str(function_res)
   }
)
# print(messages)

final_res = client.chat.completions.create(
    model="qwen3-coder-plus",
    messages=messages,
    tools=tools
)
# print(final_res)

print(final_res.choices[0].message.content)




