from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader('./LangGraph.pdf')
pages = loader.load_and_split()
print(pages[0].page_content)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True
)
paragraphs = text_splitter.create_documents([pages[0].page_content])

for para in paragraphs:
  print(para.page_content)
  print('------------------------')


