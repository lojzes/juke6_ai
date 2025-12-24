import json
import os

from fastmcp import FastMCP, Context
import psycopg2
from fastmcp.server.sampling import SamplingResult
from psycopg2 import OperationalError, ProgrammingError
from psycopg2.extras import RealDictCursor

from sys_log import SysLog
from mcp.types import TextContent
from dotenv import load_dotenv

logger = SysLog.getLogger("mysql_mcp_server")


# 获取数据库配置
def get_db_config():
    # 加载 .env 文件中的环境变量到系统环境变量中
    load_dotenv()
    # 从环境变量中获取数据库配置
    config = {
        "host": os.getenv("POSTGRESQL_HOST"),
        "user": os.getenv("POSTGRESQL_USER"),
        "password": os.getenv("POSTGRESQL_PASSWORD"),
        "database": os.getenv("POSTGRESQL_DATABASE")
    }

    # 检查是否存在配置中的关键字段
    # 记录错误信息，提示用户检查环境变量
    if not all([config["user"], config["password"], config["database"]]):
        logger.error("Missing required database configuration. Please check environment variables:")
        logger.error("MYSQL_USER, MYSQL_PASSWORD, and MYSQL_DATABASE are required")
        # 抛出一个 ValueError 异常，终止函数的执行
        raise ValueError("Missing required database configuration")

    # 配置完整，则返回包含数据库配置的字典config
    return config


# 实例化Server
mcp = FastMCP("weibo_mcp_server")

def get_db_connection():
    """创建数据库连接"""
    # 获取数据库配置
    config = get_db_config()
    connection = None
    try:
        # 连接参数：主机、端口、数据库名、用户名、密码
        connection = psycopg2.connect(
            host=config['host'],  # 数据库主机地址（本地为localhost）
            port="5432",  # PostgreSQL默认端口
            dbname=config["database"],  # 要连接的数据库名
            user=config["user"],  # 数据库用户名（默认常为postgres）
            password=config["password"]  # 数据库密码
        )
        print("数据库连接成功！")
    except OperationalError as e:
        print(f"连接失败：{e}")
    return connection


@mcp.resource("db://weibo_data/data/{date}/{limit}")
async def get_weibo_data(ctx: Context, date: str, limit: int = 10) -> str:
    """
    微博数据
    :param ctx:
    :param date:  日期，数据格式为 mmdd
    :param limit:  要求返回的数据量
    :return:  以 json格式返回数据
    """

    logger.info(f"Get weibo data from {date}")
    sql = """
        select id,user_name,publish_date,publish_time,
        weibo_content
        from t_weibo_data
        where publish_date = %s
        limit %s
    """
    with get_db_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(sql,(date,limit))
            rows = cursor.fetchall()
            #通过 Sampling 给每个微博内容增加情感倾向标签
            for row in rows:
                # 系统指令
                system_prompt = "请给出这条微博内容的情感倾向，标注分为三类的其中一个:积极，中性和消极"
                # 向 Client 发起 Sampling 请求
                response = await ctx.sample(
                    messages=row["weibo_content"],
                    system_prompt=system_prompt,
                )
                assert isinstance(response, SamplingResult)
                row['sentiment_tag'] = response.text
                print(row)
            return json.dumps(list(rows),default=str,ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="streamable-http",host="127.0.0.1", port=8000, path="/mcp")
