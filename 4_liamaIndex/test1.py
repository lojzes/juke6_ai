import os
from llama_index.core import Settings
from llama_index.llms.dashscope import DashScope,DashScopeGenerationModels
from llama_index.embeddings.dashscope import DashScopeEmbedding,DashScopeTextEmbeddingModels
from llama_index.core import  VectorStoreIndex,SimpleDirectoryReader

llm = DashScope(
    model_name=DashScopeGenerationModels.QWEN_MAX,
    api_key="sk-66bc27a6330f434f8751f8172a73064f"
)

Settings.llm = llm

embedding = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V1,
    api_key="sk-66bc27a6330f434f8751f8172a73064f"
)
Settings.embed_model = embedding

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("什么是 LangGraph")

print(response)


