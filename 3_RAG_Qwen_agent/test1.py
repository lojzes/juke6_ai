import logging
import io
import os
from qwen_agent.agents import Assistant

# 步骤 1: 配置 LLM
llm_cfg = {
    # 使用 DashScope 提供的模型服务
    'model':'qwen-max',
    'model_server':'dashscope',
    'api_key':'sk-66bc27a6330f434f8751f8172a73064f',
    'generate_cfg':{
        'top_p':0.8
    },
}

# 步骤 2：创建一个智能体，这里以'Assistant'智能体为例，它能够使用工具并读取文件

sys_instruction = ''
tools = []
files = ['./LangGraph.pdf'] #给智能体一个 PDF 文件 阅读

# 创建智能体
bot = Assistant(
    llm=llm_cfg,
    system_message=sys_instruction,
    function_list=tools,
    files=files
)
# 步骤 3: 作为聊天机器人运行智能体
messages = [] # 这里储存聊天历史
query = "langgraph是什么"
# 将用户请求添加到聊天历史
messages.append({'role':'user','content':query})
response = []
current_index = 0

# 运行智能体
for response in bot.run(messages=messages):
    current_response = response[0]['content'][current_index:]
    current_index = len(response[0]['content'])
    print(current_response,end='')

messages.extend(response)
print(messages)











