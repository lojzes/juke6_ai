
import os
class Config:
    PROMPT_TEMPLATE_TEXT_AGENT = "prompts/prompt_template_agent.txt"
    PROMPT_TEMPLATE_TEXT_GRADE = "prompts/prompt_template_grade.txt"
    PROMPT_TEMPLATE_TEXT_REWRITE = "prompts/prompt_template_rewrite.txt"
    PROMPT_TEMPLATE_TEXT_GENERATE = "prompts/prompt_template_generate.txt"

    CHROMADB_DIRECTORY = "chromadb"
    CHROMADB_COLLECTION_NAME = "demo001"

    LOG_FILE= "output/app.log"
    MAX_BYTES=5*1024*1024
    BACKUP_COUNT = 3

    # 数据库 URI，默认值
    DB_URI = os.getenv("DB_URI", "postgresql://root:root@localhost:5432/postgres?sslmode=disable")
    # openai:调用gpt模型，qwen:调用阿里通义千问大模型，oneapi:调用oneapi方案支持的模型，ollama:调用本地开源大模型
    LLM_TYPE = "qwen"
    # API服务地址和端口
    HOST ="0.0.0.0"
    PORT = 8012

