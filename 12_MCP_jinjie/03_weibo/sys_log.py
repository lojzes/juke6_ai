import logging

# 日志相关配置
logging.basicConfig(
    level=logging.INFO,
    # 新增 %(filename)s（文件名）、%(funcName)s（函数名）、%(lineno)d（行号）占位符
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s'
)


class SysLog:

    @classmethod
    def getLogger(cls,name:str):
        return logging.getLogger(name)


