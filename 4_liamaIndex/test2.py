
import json
from reprlib import recursive_repr
from pydantic import BaseModel
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter

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

from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(
    input_dir="./data",
    recursive=True,
    required_exts=[".pdf"]
)
documents = reader.load_data()

print(documents[0].text)
show_json(documents[0].json())

node_parser = TokenTextSplitter(
    chunk_size=512,#每个 chunk 的最大长度
    chunk_overlap=200# chunk 之间重叠长度
)

nodes = node_parser.get_nodes_from_documents(documents,show_progress=True)

show_json(nodes[1].json())
show_json(nodes[2].json())



