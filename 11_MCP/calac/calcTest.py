
# ClientSession 表示客户端会话，用于域服务器交互
# StdioServerParameters 定义与服务器的 stdio 链接参数
# stio_client 提供与服务器的 stdio连接上下文管理器

import os
from xml.dom import DOMException

from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
# 为 stdio 链接创建服务器参数
server_parameters = StdioServerParameters(
    #服务器执行的命令 本地 server 方式运行
    command="python",
    # 启动命令的附加参数，这里是运行 example_server.py
    args=["calculatorMCPServer.py"],
    # 环境变量，默认为 None ,表示使用当前的环境表里
    env=None,
)
# 服务器端功能测试
async def run():
    async with stdio_client(server_parameters) as (reader, writer):
        #创建一个客户端会话对象，通过 read 和 write 流与服务器交互
        async with ClientSession(reader,writer) as session:
            try:
                # 向服务器发送初始化请求，确保连接准备就行
                # 建立初始状态，并让服务器返回其功能和版本信息
                capabilities = await session.initialize()
                print(f"Support capabilities: {capabilities.capabilities} /n/n")

                # tools = await session.list_tools()
                # print(f"supported tools: {tools}/n/n")
                # with open("output.txt","w",encoding="utf-8") as f:
                #     f.write(str(tools))
                # 相关功能测试
                add = await session.call_tool("add",{"a":1,"b":2})
                print(f"add:{add}")
                print(f"add:{add.content[0].text}")
                subtract = await session.call_tool("subtract", {"a": 1, "b": 2})
                print(f"subtract:{subtract}")
                print(f"subtract:{subtract.content[0].text}")
                multity = await session.call_tool("multity", {"a": 1, "b": 2})
                print(f"multity:{multity}")
                print(f"multity:{multity.content[0].text}")
                divide = await session.call_tool("divide", {"a": 1, "b": 2})
                print(f"divide:{divide}")
                print(f"divide:{divide.content[0].text}")
            except DOMException as e:
                print(e)


if __name__ == "__main__":
    asyncio.run(run())