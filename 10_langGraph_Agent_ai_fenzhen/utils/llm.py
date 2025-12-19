import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI

import logging
from langchain_openai import OpenAIEmbeddings

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "qwen":{
        "base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key":os.getenv("Ali_KEY"),
        "chat_model":"qwen-max",
        "embedding_model":"text-embedding-v1",
    }
}

DEFAULT_LLM_TYPE = "qwen"
DEFAULT_TEMPERATURE = 0.5

class LLMInitializationError(Exception):
    pass

def initialize_llm(llm_type, temperature=DEFAULT_TEMPERATURE)->tuple[ChatOpenAI,OpenAIEmbeddings]:
    """
    初始化 LLM 实例

    :param llm_type: LLM 类型
    :param temperature:
    :return: ChatOpenAI 初始化后的 LLM 实例
    :raise: LLMInitializationError 当 LLM 初始化失败时抛出
    """
    try:
        if llm_type not in MODEL_CONFIGS:
            raise ValueError(f"不支持的 LLM类型 {llm_type},可用类型 {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[llm_type]
        #创建LLM实例
        llm_chat = ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["chat_model"],
            temperature=temperature,
            timeout=30, #单位秒
            max_retries=2
        )
        llm_embeddings = OpenAIEmbeddings(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["embedding_model"],
            deployment=config["embedding_model"],
            check_embedding_ctx_length=False
        )
        logger.info(f"成功初始化{llm_type} LLM")
        return llm_chat, llm_embeddings
    except ValueError as ve:
        logger.error(f"LLM配置错误 {str(ve)}")
        raise LLMInitializationError(f"LLM 配置有误 {str(ve)}")
    except LLMInitializationError as e:
        logger.error(f"初始化LLM失败 {str(e)}")
        raise LLMInitializationError(e)


def get_llm(llm_type, temperature=DEFAULT_TEMPERATURE)->ChatOpenAI:
    try:
        return initialize_llm(llm_type)
    except LLMInitializationError as e:
        logger.warn(f"使用默认配置重试 {str(e)}")
        if llm_type != DEFAULT_LLM_TYPE:
            return initialize_llm(DEFAULT_LLM_TYPE)
        raise


# 示例使用
if __name__ == "__main__":
    try:
        llm_qwen = get_llm("qwen")
    except LLMInitializationError as e:
        print(e)





