from openai.types import Embedding
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams,Distance
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,get_response_synthesizer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core.postprocessor import LLMRerank,SimilarityPostprocessor
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.llms.dashscope import DashScope,DashScopeGenerationModels
from llama_index.embeddings.dashscope import DashScopeEmbedding,DashScopeTextEmbeddingModels

EMBEDDING_DIM=1536
COLLECTION_NAME="full_demo"
PATH="./qdrant_db"

client = QdrantClient(path=PATH)


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

# 2 指定全局文档处理的 Ingestion Pipeline
Settings.transformations = [SentenceSplitter(chunk_size=512,chunk_overlap=200)]
# 3 加载本地文档
documents = SimpleDirectoryReader("./data").load_data()

if client.collection_exists(collection_name=COLLECTION_NAME):
    client.delete_collection(collection_name=COLLECTION_NAME)

# 4 创建 collection
client.create_collection(COLLECTION_NAME,
                         vectors_config=VectorParams(
                           size=EMBEDDING_DIM,distance=Distance.COSINE
                         )
                )

# 5 创建 Vector Store
vectore_store = QdrantVectorStore(
    client=client,collection_name=COLLECTION_NAME,
)
# 6 指定 Vectore Store 的 Store用于 index
storage_context = StorageContext.from_defaults(vector_store=vectore_store)
index = VectorStoreIndex.from_documents(documents,storage_context=storage_context)

# 7定义检索后排序模型
reranker = LLMRerank(top_n=2)
# 最终打分低于 0.6的文档被过滤掉
sp = SimilarityPostprocessor(similarity_cutoff=0.6)

# 8 定义 Rag fusion 检索器
fusion_retriever = QueryFusionRetriever(
    [index.as_retriever()],
    similarity_top_k=5 ,#检索召回 top key 结果
    num_queries=3,#生成 query 数
    use_async=False,
    query_gen_prompt="",#可以自定义 query 生成prompt 模板
)

# 9 构建单论 query engine
query_engine = RetrieverQueryEngine.from_args(
    retriever=fusion_retriever,
    node_postprocessors=[reranker,sp],
    response_synthesizer=get_response_synthesizer(
        response_mode=ResponseMode.REFINE
    )
)
# 10 对话引擎
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    #condense_question_prompt="",#可自定义 chat message prompt
)

# 测试多轮对话
# User :  LangGraph 是什么
while True:
    question = input("User: ")
    if question.strip() == "":
        break
    response = chat_engine.chat(question)
    print(response)
















