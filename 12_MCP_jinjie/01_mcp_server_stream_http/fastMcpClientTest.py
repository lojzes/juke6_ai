import asyncio

from fastmcp import Client

async def main():
    async with Client("http://127.0.0.1:8000/mcp") as mcp_client:
        tools = await mcp_client.list_tools()
        print(f"\nTools: {tools}")
        # 调用方法
        re = await mcp_client.call_tool("add",{"a":1,"b":2})
        print(re.content[0].text)
        results = await mcp_client.list_resources()
        print(f"\nresults: {results}")
        # 读取资源
        res = await mcp_client.read_resource("db://tables")
        print(f"\nres: {res}")
        # 带参数的资源
        results = await mcp_client.list_resource_templates()
        print(f"\nlist_resource_templates: {results}")
        # 调用带参数的资源
        res = await mcp_client.read_resource("db://tables/china_province/data/10")
        print(f"\nres: {res}")
        prompts = await mcp_client.list_prompts()
        print(f"\nprompts: {prompts}")
        # 获取 prompt
        res = await mcp_client.get_prompt("induce_china_province",{"province":"北京"})
        # Access the personalized messages
        for message in res.messages:
            print(f"message: {message.content}")
if __name__ == "__main__":
    asyncio.run(main())



