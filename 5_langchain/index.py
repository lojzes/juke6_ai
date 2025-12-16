
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader('./LangGraph.pdf')
pages = loader.load_and_split()
# print(pages[0].page_content)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True
)
texts = text_splitter.create_documents(
    [page.page_content for page in pages[:1]]
)
# 灌库
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key="sk-66bc27a6330f434f8751f8172a73064f"
)
index = FAISS.from_documents(texts,embeddings)
# 检索top-5
retriever = index.as_retriever(search_kwargs={"k":2})

docs = retriever.invoke("langGraph 是什么")

for doc in docs:
    print(doc.page_content)
    print("-------")
