import json
import os

from mcp.server import FastMCP
import psycopg2
from psycopg2 import OperationalError, ProgrammingError
from psycopg2.extras import RealDictCursor

from sys_log import SysLog
from mcp.types import Resource, Tool, TextContent
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
mcp = FastMCP("mysql_mcp_server")

def get_db_connection():
    """创建数据库连接"""
    # 获取数据库配置
    config = get_db_config()
    connection = None
    try:
        # 连接参数：主机、端口、数据库名、用户名、密码
        connection = psycopg2.connect(
            host=config['host'],  # 数据库主机地址（本地为localhost）
            port="5432",       # PostgreSQL默认端口
            dbname=config["database"],  # 要连接的数据库名
            user=config["user"],    # 数据库用户名（默认常为postgres）
            password=config["password"] # 数据库密码
        )
        print("数据库连接成功！")
    except OperationalError as e:
        print(f"连接失败：{e}")
    return connection

conn = get_db_connection()
print(conn)