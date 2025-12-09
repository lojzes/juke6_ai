
import json
from reprlib import recursive_repr

from llama_index.core.base.embeddings.base import similarity
from llama_index.core.node_parser import TokenTextSplitter
from openai import vector_stores
from pydantic import BaseModel

from test1 import documents


def show_json(data):
    """
    用于展示 json 数据
    """
    if isinstance(data, str):
        obj = json.loads(data)
        print(json.dumps(obj, indent=4,ensure_ascii=False))
    elif isinstance(data, dict):
        print(json.dumps(data, indent=4, ensure_ascii=False))
    elif issubclass(type(data), BaseModel):
        print(json.dumps(data.dict, indent=4, ensure_ascii=False))

def show_list_obj(data):
    """
     用于展示一组对象
    """
    if isinstance(data, list):
        for item in data:
           show_list_obj(data)
    else:
        raise ValueError("Input is not a list")

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

reader = SimpleDirectoryReader(
    input_dir="./data",
    recursive=True,
    required_exts=[".pdf"]
)
documents = reader.load_data()

node_parser = TokenTextSplitter(
    chunk_size=512,chunk_overlap=200
)

nodes = node_parser.get_nodes_from_documents(documents,show_progress=True)

# 构建 index 默认是在内存中
index = VectorStoreIndex(nodes)

vector_retriver = index.as_retriever(
     similarity_top_k = 2
)
# 检索
result = vector_retriver.retrieve("LangGraph是什么")

print(result[0].text)