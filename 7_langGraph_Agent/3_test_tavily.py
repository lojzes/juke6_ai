import json
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_tavily import TavilySearch

tavily_tool = TavilySearch(
    tavily_api_key=os.getenv("Tavily_KEY"),
    max_results=2)

tools = [tavily_tool]

result = tavily_tool.invoke("什么是 LangGraph ?")

for res in result['results']:
    print(res['content'])
