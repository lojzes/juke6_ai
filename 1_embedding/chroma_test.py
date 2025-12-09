import chromadb

# 临时
# client = chromadb.EphemeralClient()
# print(client)

# 持久化运行
client = chromadb.PersistentClient('./chromadb')
print(client)


from chromadb.utils import embedding_functions
# 默认情况下，Chroma 使用 DefaultEmbeddingFunction，它是基于 Sentence Transformers 的 MiniLM-L6-v2 模型
default_ef = embedding_functions.DefaultEmbeddingFunction()
# 使用 OpenAI 的嵌入模型，默认使用 text-embedding-ada-002 模型
#openai_ef = embedding_functions.OpenAIEmbeddingFunction(
#    api_key="YOUR_API_KEY",
#    model_name="text-embedding-3-small"
# )

# 没有则创建后返回、否则直接返回
collection = client.get_or_create_collection(
    name = "my_collection",
    configuration = {
    #HNSW 索引算法，基于图的近似最近邻搜索算法（Approximate Nearest Neighbor, ANN）
    "hnsw": {
        "space": "cosine", # 指定余弦相似度计算
        "ef_search": 100,
        "ef_construction": 100,
        "max_neighbors": 16,
        "num_threads": 4
    },
    #指定向量模型
    "embedding_function": default_ef
    }
)
collection = client.get_collection(name="my_collection")
print(collection)
#  返回前 10条
print(collection.peek(limit=10))

result = collection.query(
     query_texts=["RAG 是什么？"],
     n_results=3,
     # where={"source":"RAG"},
     # where_document={"$contains":"检索增强生成"}
)
print(result)