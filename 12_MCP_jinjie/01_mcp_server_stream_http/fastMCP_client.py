import asyncio
import json
import logging
import os

import mcp

from sys_log import SysLog
from typing import Dict, List, Optional, Any
import requests
from dotenv import load_dotenv
from fastmcp import FastMCP, Client
from openai import OpenAI

# 配置日志记录
logger = SysLog.getLogger(__name__)


# 用于管理MCP客户端的配置和环境变量
class Configuration:
    # 初始化对象，并加载环境变量
    def __init__(self) -> None:
        # 加载环境变量（通常从 .env 文件中读取）
        self.load_env()
        self.base_url = os.getenv("LLM_Ali_BASE_URL")
        self.api_key = os.getenv("Ali_KEY")
        self.chat_model = os.getenv("LLM_Ali_CHAT_MODEL")

    # @staticmethod，表示该方法不依赖于实例本身，可以直接通过类名调用
    @staticmethod
    def load_env() -> None:
        load_dotenv()

    # 从指定的 JSON 配置文件中加载配置
    # file_path: 配置文件的路径
    # 返回值: 一个包含配置信息的字典
    # FileNotFoundError: 文件不存在时抛出
    # JSONDecodeError: 配置文件不是有效的 JSON 格式时抛出
    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        # 打开指定路径的文件，以只读模式读取
        # 使用 json.load 将文件内容解析为 Python 字典并返回
        with open(file_path, 'r') as f:
            return json.load(f)

    # @property，将方法转换为只读属性，调用时不需要括号
    # 提供获取 llm_api_key 的接口
    @property
    def llm_api_key(self) -> str:
        # 检查 self.api_key 是否存在
        if not self.api_key:
            # 如果不存在，抛出 ValueError 异常
            raise ValueError("LLM_API_KEY not found in environment variables")
        # 返回 self.api_key 的值
        return self.api_key

    # @property，将方法转换为只读属性，调用时不需要括号
    # 提供获取 llm_base_url 的接口
    @property
    def llm_base_url(self) -> str:
        # 检查 self.base_url 是否存在
        if not self.base_url:
            # 如果不存在，抛出 ValueError 异常
            raise ValueError("LLM_BASE_URL not found in environment variables")
        # 返回 self.base_url 的值
        return self.base_url

    # @property，将方法转换为只读属性，调用时不需要括号
    # 提供获取 llm_chat_model 的接口
    @property
    def llm_chat_model(self) -> str:
        # 检查 self.base_url 是否存在
        if not self.chat_model:
            # 如果不存在，抛出 ValueError 异常
            raise ValueError("LLM_CHAT_MODEL not found in environment variables")
        # 返回 self.base_url 的值
        return self.chat_model


# 管理与LLM的通信
class LLMClient:
    def __init__(self, base_url: str, api_key: str, chat_model: str) -> None:
        self.base_url: str = base_url
        self.api_key: str = api_key
        self.chat_model: str = chat_model

    # 向 LLM 发送请求，并返回其响应
    # messages: 一个字典列表，每个字典包含消息内容，通常是聊天对话的一部分
    # 返回值: 返回 LLM 的响应内容，类型为字符串
    # 如果请求失败，抛出 RequestException
    def get_response(self, messages: List[Dict[str, str]]) -> str:
        # 指定 LLM 提供者的 API 端点，用于发送聊天请求
        client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
            api_key=self.api_key,
            base_url=self.base_url,
        )
        completion = client.chat.completions.create(
            # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            model=self.chat_model,
            messages=messages
        )
        try:
            # 从 JSON 响应中提取工具的输出内容
            return completion.choices[0].message.content
        # 如果请求失败（如连接错误、超时或无效响应等），捕获 RequestException 异常
        # 记录错误信息，str(e) 提供异常的具体描述
        except requests.exceptions.RequestException as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logger.error(error_message)
            # 如果异常中包含响应对象（e.response），进一步记录响应的状态码和响应内容
            # 这有助于分析请求失败的原因（例如服务端错误、API 限制等）
            if e.response is not None:
                status_code = e.response.status_code
                logger.error(f"Status code: {status_code}")
                logger.error(f"Response details: {e.response.text}")
            # 返回一个友好的错误消息给调用者，告知发生了错误并建议重试或重新措辞请求
            return f"I encountered an error: {error_message}. Please try again or rephrase your request."


# 代表各个资源及其属性和格式
class Resource:
    # 构造函数，在类实例化时调用
    # uri: 表的唯一资源标识符
    # name: 资源的名称
    # mimeType: MIME 类型，表示资源的数据类型
    # description: 描述信息
    def __init__(self, uri: str, name: str, description: str, mimeType: str) -> None:
        self.uri: str = uri
        self.name: str = name
        self.description: str = description
        self.mimeType: str = mimeType

    # 将资源的信息格式化为一个字符串，适合语言模型（LLM）使用
    def format_for_llm(self) -> str:
        return f"""
                URI: {self.uri}
                Name: {self.name}
                Description: {self.description}
                MimeType: {self.mimeType}
                """


# 代表各个工具及其属性和格式
class Tool:
    # 构造函数，在类实例化时调用
    # name: 工具的名称
    # description: 工具的描述信息
    # input_schema: 工具的输入架构，通常是一个描述输入参数的字典
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: Dict[str, Any] = input_schema

    # 将工具的信息格式化为一个字符串，适合语言模型（LLM）使用
    # 返回值: 包含工具名称、描述和参数信息的格式化字符串
    def format_for_llm(self) -> str:
        args_desc = []
        if 'properties' in self.input_schema:
            for param_name, param_info in self.input_schema['properties'].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                if param_name in self.input_schema.get('required', []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
                Tool: {self.name}
                Description: {self.description}
                Arguments:
                {chr(10).join(args_desc)}
                """


# 协调用户、 LLM和工具之间的交互
class ChatSession:
    # servers: 一个 Server 类的列表，表示多个服务器的实例
    # llm_client: LLMClient 的实例，用于与 LLM 进行通信
    def __init__(self, mcp_client: Client, llm_client: LLMClient) -> None:
        self.mcp_client: Client = mcp_client
        self.llm_client: LLMClient = llm_client

    # 负责处理从 LLM 返回的响应，并在需要时执行工具
    async def process_llm_response(self, llm_response: str) -> str:
        try:
            # 尝试将 LLM 响应解析为 JSON ，以便检查是否包含 tool 和 arguments 字段
            llm_call = json.loads(llm_response)

            # 1、如果响应包含工具名称和参数，执行相应工具
            if "tool" in llm_call and "arguments" in llm_call:
                logger.info(f"Executing tool: {llm_call['tool']}")
                logger.info(f"With arguments: {llm_call['arguments']}")
                # 遍历每个服务器，检查是否有与响应中的工具名称匹配的工具
                try:
                    # 执行工具
                    tool_result = await self.mcp_client.call_tool(llm_call['tool'], llm_call['arguments'])
                    logger.info(f"Tool execution result: {tool_result.content[0].text}")
                    return f"Tool execution result: {tool_result.content[0].text}"
                except Exception as e:
                    logger.error(f" does not have 'list_tools' method: {e}")
                return f"No server found with tool: {llm_call['tool']}"

            # 2、如果响应包含资源名称和URI，执行读取相应的资源
            if "resource" in llm_call:
                logger.info(f"Executing resource name: {llm_call['resource']}")
                logger.info(f"Executing resource description: {llm_call['description']}")
                logger.info(f"With URI: {llm_call['uri']}")
                # 遍历每个服务器，检查是否有与响应中的资源名称匹配的资源
                try:
                    # 读取资源
                    result = await self.mcp_client.read_resource(llm_call["uri"])
                    # 返回资源的执行结果
                    logger.info(f"Resource execution result: {result}")
                    return f"Resource execution result: {result}"
                except Exception as e:
                    logger.error(f"Server does not have 'list_resources' method: {e}")
                return f"No server found with resource: {llm_call['uri']}"
            return llm_response

        # 如果响应无法解析为 JSON，返回原始 LLM 响应
        except json.JSONDecodeError:
            return llm_response

    # 方法: start 用于启动整个聊天会话，初始化服务器并开始与用户的互动
    async def start(self) -> None:
        try:
            # 遍历所有服务器，调用 list_tools() 获取每个服务器的工具列表
            all_tools = []
            try:
                tools = await self.mcp_client.list_tools()
                all_tools.extend([Tool(tool.name, tool.description, tool.inputSchema) for tool in tools])
                tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
                logger.info(f"tools_description {tools_description}")
            except Exception as e:
                print(f"Error while formatting resources for LLM: {e}")
                tools_description = ""

            # # 遍历所有服务器，调用 list_resources() 获取每个服务器的资源列表
            all_resources = []
            resources = await self.mcp_client.list_resources()
            for resource in resources:
                all_resources.append(
                    Resource(resource.uri.unicode_string(), resource.name, resource.description, resource.mimeType))

            resource_templates = await self.mcp_client.list_resource_templates()
            for resource in resource_templates:
                all_resources.append(Resource(resource.uriTemplate, resource.name, resource.description, resource.mimeType))

            # 将所有资源的描述信息汇总，生成供 LLM 使用的资源描述字符串
            try:
                resources_description = "\n".join([resource.format_for_llm() for resource in all_resources])
                logger.info(f"resources_description {resources_description}")
            except Exception as e:
                print(f"Error while formatting resources for LLM: {e}")
                resources_description = ""

            # 构建一个系统消息，作为 LLM 交互的指令，告知 LLM 使用哪些工具以及如何与用户进行交互
            # 系统消息强调 LLM 必须以严格的 JSON 格式请求工具，并且在工具响应后将其转换为自然语言响应
            system_message = f"""你是一名得力助手，可使用以下资源和工具：

                            资源：{resources_description}
                            工具：{tools_description}
                            
                            根据用户问题选择合适的资源或工具。若无需使用任何资源或工具，直接回复即可。

                            重要提示：当需要使用资源时，仅需以以下指定的JSON对象格式响应，不可包含其他内容：
                            {{
                                "resource": "资源名称",
                                "description": "资源描述",
                                "uri": "资源统一资源标识符"
                            }}

                            收到资源的响应后：
                            1. 将原始数据转化为自然、口语化的回应
                            2. 回应需简洁但信息完整
                            3. 聚焦最相关的信息
                            4. 结合用户问题的上下文语境
                            5. 避免直接重复原始数据

                            重要提示：当需要使用工具时，仅需以以下指定的JSON对象格式响应，不可包含其他内容：
                            {{
                                "tool": "工具名称",
                                "description": "工具描述",
                                "arguments": {{
                                    "参数名称": "参数值"
                                }}
                            }}

                            收到工具的响应后：
                            1. 将原始数据转化为自然、口语化的回应
                            2. 回应需简洁但信息完整
                            3. 聚焦最相关的信息
                            4. 结合用户问题的上下文语境
                            5. 避免直接重复原始数据

                            请仅使用上述明确指定的资源或工具。"""
            # 消息初始化 创建一个消息列表，其中包含一个系统消息，指示 LLM 如何与用户交互
            messages = [
                {
                    "role": "system",
                    "content": system_message
                }
            ]

            # 交互循环
            while True:
                try:
                    # 等待用户输入，如果用户输入 quit 或 exit，则退出
                    user_input = input("user: ").strip().lower()
                    if user_input in ['quit', 'exit']:
                        logging.info("\nExiting...")
                        break

                    # 将用户输入添加到消息列表
                    messages.append({"role": "user", "content": user_input})

                    # 调用 LLM 客户端获取 LLM 的响应
                    llm_response = self.llm_client.get_response(messages)
                    logging.info("\nAssistant: %s", llm_response)

                    # 调用 process_llm_response 方法处理 LLM 响应，执行工具（如果需要）
                    result = await self.process_llm_response(llm_response)

                    # 如果工具执行结果与 LLM 响应不同，更新消息列表并再次与 LLM 交互，获取最终响应
                    if result != llm_response:
                        messages.append({"role": "assistant", "content": llm_response})
                        messages.append({"role": "system", "content": result})

                        final_response = self.llm_client.get_response(messages)
                        logging.info("\nFinal response: %s", final_response)
                        messages.append({"role": "assistant", "content": final_response})
                    else:
                        messages.append({"role": "assistant", "content": llm_response})

                # 处理 KeyboardInterrupt，允许用户中断会话
                except KeyboardInterrupt:
                    logger.info("\nExiting...")
                    break
        # 清理资源: 无论会话如何结束（正常结束或由于异常退出），都确保调用 cleanup_servers 清理服务器资源
        finally:
            print("\n---------------------")


async def main() -> None:
    # 创建一个 Configuration 类的实例
    config = Configuration()
    # 调用 Configuration 类中的 load_config 方法来加载配置文件 servers_config.json
    # server_config = config.load_config('servers_config.json')
    # 遍历 server_config['mcpServers'] 字典，并为每个服务器创建一个 Server 实例，传入服务器名称 (name) 和配置信息 (srv_config)
    # 结果是一个 Server 对象的列表，保存在 servers 变量中
    # servers = [Server(name, srv_config) for name, srv_config in server_config['mcpServers'].items()]
    async with Client("http://127.0.0.1:8000/mcp") as mcp_client:
        # 创建一个 LLMClient 实例，用于与 LLM (大语言模型) 进行交互
        llm_client = LLMClient(config.llm_base_url, config.llm_api_key, config.llm_chat_model)
        # 创建一个 ChatSession 实例，负责管理与用户的聊天交互、LLM 响应和工具执行
        # 将之前创建的 servers 和 llm_client 传递给 ChatSession 构造函数，初始化会话
        chat_session = ChatSession(mcp_client, llm_client)
        # 调用 ChatSession 类的 start 方法，启动聊天会话
        # 由于 start 是一个异步方法，所以使用 await 等待该方法执行完毕
        # start 方法将处理用户的输入、与 LLM 交互、执行工具（如果需要）等，并持续运行直到用户选择退出
        await chat_session.start()


# 测试内容 执行 streamablehttpServer.py 和 clientChatTest.py
# 有哪些表可以使用
# 查询学生表中的数据
# 查询学生成绩表表中的数据
# 查询学生成绩表表中成绩最高的数据
# 对学生信息表和学生成绩表进行联表查询，生成每个学生的学生姓名、成绩
# 将学生姓名为王明的改为小明，并返回最新的信息表
if __name__ == "__main__":
    asyncio.run(main())
