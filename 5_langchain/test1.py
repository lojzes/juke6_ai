from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import(
PromptTemplate,HumanMessagePromptTemplate
)
from pydantic import BaseModel ,Field

llm = init_chat_model('deepseek-chat',model_provider='deepseek',
        api_key= "sk-fa1e532719fc46aa83166d15619bb76a"
    )
# print(model)
# response = model.invoke('你是谁')
# print(response.content)

from langchain_core.messages import(
    AIMessage, # 等价于 OpenAI 接口中的 assisant role
    HumanMessage, # 等价于 OpenAI 接口中的 user role
    SystemMessage # 等价于 OpenAI 接口中的 system role
)

messages = [
    SystemMessage(content="你是 lozes 大模型课程助理。"),
    HumanMessage(content="我是 lojzes 大模型学员，我叫 lojzes"),
    AIMessage(content="欢迎！"),
    HumanMessage(content="我是谁？")
]

response = llm.invoke(messages)
print(response.content)

# 定义输出答案
class Date(BaseModel):
    year:int = Field(description="Year")
    month:int = Field(description="Month")
    day:int = Field(description="Day")
    ear:int = Field(description="BC or AD")

query="2025年12月11日的天气晴"
parser = JsonOutputParser(pydantic_object=Date)

prompt = PromptTemplate(
    template="提取用户输入中的日期。\n用户输入:{query}\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)
input_prompt = prompt.format_prompt(query=query,)
output = llm.invoke(input_prompt)
print("原始输出:\n" + output.content)
print("\n 解析后：")
print(parser.invoke(output))





