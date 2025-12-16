import os
from dotenv import load_dotenv
load_dotenv()
import re
import logging
import sys

from llama_cloud.types import retriever
from llama_index.core import (
 PromptTemplate,Settings,SimpleDirectoryReader,VectorStoreIndex,load_index_from_storage,
StorageContext,QueryBundle
)
from llama_index.core.base.embeddings.base import similarity
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.chat_engine.utils import get_response_synthesizer
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import MetadataMode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.dashscope import DashScope,DashScopeGenerationModels
from llama_index.embeddings.dashscope import DashScopeEmbedding,DashScopeTextEmbeddingModels

# 查看底层细节，要打开 debug日志
logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def remove_english( input_file,output_file):
    """
    删除文件中所有英文字符生成新文件
    :param input_file:
    :param output_file:
    :return:
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f_in:
            content = f_in.read()
            # 使用正则表达式移除所有英文字母
            filtered_content = re.sub('[A-Za-z/]', '', content)
            with open(output_file, "w", encoding="utf-8") as f_out:
                f_out.write(filtered_content)
            print(f'处理完成，已经生成新文件')

    except Exception as e:
        print(f'处理出错:{str(e)}')


SYSTEM_PROMPT = """
You are helpfull AI assistant.
"""
query_wrapper_prompt = PromptTemplate(
   "[INST]<<SYS>>\n" + SYSTEM_PROMPT +"<</SYS>>\n\n{query_str}[/INST]"
)

llm = DashScope(
    model_name=DashScopeGenerationModels.QWEN_MAX,
    api_key=os.getenv("API_KEY"),
)
Settings.llm = llm
embedding = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V1,
    api_key=os.getenv("API_KEY")
)
Settings.embed_model = embedding
# 查看底层原理
# 使用 LlamaDebugHandler 构建事件回溯器，以追踪 LlamaIndex执行过程中发生的事件
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manger = CallbackManager([llama_debug])
Settings.callback_manager = callback_manger


documents = SimpleDirectoryReader("./data").load_data()
# 文档切分
# index = VectorStoreIndex.from_documents(documents,transformations=[SentenceSplitter(chunk_size=256)])
# 向量存储,不设置的话，是存储在内存中
# 将 embedding向量和向量索引存储在文件中
#  是存储路径
db_dir = "./doc_emb_db"
# index.storage_context.persist(persist_dir='./doc_emb_db')
#  从本地文件中读取 embedding向量和向量索引
storage_context = StorageContext.from_defaults(persist_dir=db_dir)
#根据存储的 embedding向量和向量索引重新构建检索索引
index = load_index_from_storage(storage_context=storage_context)
# 构建查询引擎
# streaming 流式输出
# similarity_top_k 检索结果的数量
# query_engine = index.as_query_engine(streaming=True,similarity_top_k=5)
# 构建 response synthesizer
# 构建 retrivever

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
)

response_synthesizer = get_response_synthesizer(
    llm=llm,
    callback_manager=callback_manger,
    qa_messages=[ChatMessage],
    refine_messages=[ChatMessage]
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)]
)

# 未完成
# 1. 使用自定义 Prompt
# 2. 使用ragas 进行评测

# 追踪哪些片段被检索
# 获取我们抽取出的相似度 top 5 的片段
contexts = query_engine.retrieve(
    QueryBundle("langGraph是什么")
)
print('-'*10 + 'ref' + '-'*10)
for i,context in enumerate(contexts):
    print('#'*10 + f'chunk {i} start' + '#'*10)
    content = context.node.get_content(metadata_mode=MetadataMode.LLM)
    print(content)
    print('#' * 10 + f'chunk {i} end' + '#' * 10)

# 查询
response = query_engine.query("langGraph是什么")
response.print_response_stream()