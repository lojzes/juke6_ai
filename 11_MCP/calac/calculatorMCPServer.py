import logging
from mcp.server import FastMCP
from mcp.types import TextContent

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )

logger = logging.getLogger(__name__)
# 初始化 FastMCP 服务器，指定服务器名称 calculate
mcp = FastMCP("calculator")
# 定义加法工具函数
@mcp.tool()
async def add(a:float, b:float) -> list[TextContent]:
    """执行加法运算

    Args:
        a (float): 第一个参数
        b (float): 第二个参数
    """
    logger.info(f"add: {a}, {b}")
    result = a + b
    return [TextContent(type="text",text=str(result))]

@mcp.tool()
async def subtract(a:float, b:float) -> list[TextContent]:
    """执行减法运算

    Args:
        a (float): 第一个参数
        b (float): 第二个参数
    """
    logger.info(f"add: {a}, {b}")
    result = a - b
    return [TextContent(type="text",text=str(result))]

@mcp.tool()
async def multity(a:float, b:float) -> list[TextContent]:
    """执行乘法运算

    Args:
        a (float): 第一个参数
        b (float): 第二个参数
    """
    logger.info(f"add: {a}, {b}")
    result = a * b
    return [TextContent(type="text",text=str(result))]

@mcp.tool()
async def divide(a:float, b:float) -> list[TextContent]:
    """执行除法运算

    Args:
        a (float): 第一个参数
        b (float): 第二个参数
    """
    logger.info(f"add: {a}, {b}")
    result = a / b
    return [TextContent(type="text",text=str(result))]


if __name__ == "__main__":
    logger.info(f"calculator: 启动成功")
    # 初始化并运行 FastMCP 服务器，使用标准输入输出作为传输方式
    mcp.run()

