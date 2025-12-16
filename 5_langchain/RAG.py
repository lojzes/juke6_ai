
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.prompts import ChatPromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_classic.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

loader = PyMuPDFLoader('./LangGraph.pdf')
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True
)
texts = text_splitter.create_documents(
    [page.page_content for page in pages[:1]]
)
llm = init_chat_model('deepseek-chat',model_provider='deepseek',
        api_key= "sk-fa1e532719fc46aa83166d15619bb76a"
    )
# 灌库
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key="sk-66bc27a6330f434f8751f8172a73064f"
)
index = FAISS.from_documents(texts,embeddings)
# 检索top-5
retriver = index.as_retriever(search_kwargs={"k":2})

template = """ Answer the question base only on the following context:
{context}

Question:{question}
"""

prompt = ChatPromptTemplate.from_template(template)
#Chain
rag_chain = (
    {"question":RunnablePassthrough(),"context":retriver}
    | prompt
    | llm
    | StrOutputParser()
)

out = rag_chain.invoke("LangGraph 是什么？")
print(out)



