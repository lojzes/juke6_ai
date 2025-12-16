import os
from dotenv import load_dotenv
load_dotenv()
import requests
import json
from typing import  TypedDict
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCall

class FunctionCallingResult(TypedDict):
    name: str
    arguments: str

class ModelRequestWithFunctionCalling:
    def __init__(self):
        self._client = OpenAI(
            api_key=os.getenv("Ali_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self._function_infos = {}
        self._function_mappings = {}
        self._messages = []
    def register_function(self,*,name,description,parameters,function,**kwargs):
        self._function_infos.update(
            {
                name:{
                    "type":"function",
                    "function":{
                        "name":name,
                        "description":description,
                        "parameters":parameters,
                        **kwargs,
                    },
                }
            }
        )
        self._function_mappings.update({name:function})
        return self

    def reset_messages(self):
        self._messages = []
        return self

    def append_message(self,role,content,tool_calls=None,**kwargs):
        msg = {
            "role": role,
            "content": content or "",  # 确保content不为None
            **kwargs,
        }
        # 仅当tool_calls存在且非空时添加，且转为字典格式
        if tool_calls:
            msg["tool_calls"] = [self._tool_call_to_dict(tc) for tc in tool_calls]
        self._messages.append(msg)
        print("[Processing Messages]:", self._messages)
        return self
    # 辅助方法：将ToolCall对象转为可序列化的字典
    def _tool_call_to_dict(self, tool_call: ChatCompletionMessageToolCall) -> dict:
        return {
            "id": tool_call.id,
            "type": tool_call.type,
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            }
        }
    def _call(self,function_call_result: FunctionCallingResult):
        function = self._function_mappings[function_call_result.name]
        arguments = json.loads(function_call_result.arguments)
        return function(**arguments)
    def request(self,*,role="user",content=None):
        if role and content:
            self._messages.append({
                "role":role,
                "content":content,
            })
        result = self._client.chat.completions.create(
            model="qwen3-coder-plus",
            messages=self._messages,
            tools=self._function_infos.values()
        )
        self.append_message(**dict(result.choices[0].message))
        if result.choices[0].message.tool_calls:
            for tool_call in result.choices[0].message.tool_calls:
                call_result = self._call(tool_call.function)
                self.append_message("tool",str(call_result),tool_call_id=tool_call.id)
                return self.request()
        else:
            self.append_message("assistant",result.choices[0].message.content)
            return result.choices[0].message.content


amap_key = os.getenv("Gaode_KEY")
amap_url="https://restapi.amap.com/v5"

def get_loaction_coordinate(location,city):
    url = f"{amap_url}/place/text?key={amap_key}&keywords={location}&region={city}"
    r = requests.get(url)
    result = r.json()
    if "pois" in result and result["pois"]:
        return result["pois"][0]
    return None

def search_nearby_pois(longitude,latitude,keyword):
    url = f"{amap_url}/place/around?key={amap_key}&keywords={keyword}&location={longitude},{latitude}"
    r = requests.get(url)
    result = r.json()
    ans = ""
    if "pois" in result["pois"] and result["pois"]:
        for i in range(min(3,len(result["pois"]))):
            name = result["pois"][i]["name"]
            address = result["pois"][i]["address"]
            distance = result["pois"][i]["distance"]
            ans += f"{name}\n {address}\n 距离 :{distance}米 \n\n"

    return ans

function_calling_request = ModelRequestWithFunctionCalling()

function_calling_request.register_function(
        name="get_loaction_coordinate",
        description="根据 POI 名称,获得 POI的经纬度坐标",
        parameters={
            "type":"object",
            "properties":{
                "location":{
                    "type":"string",
                    "description":"POI名称，必须是中文"
                },
                "city":{
                    "type":"string",
                    "description":"POI所在的城市，必须是中文"
                }
            },
            "required":["location","city"]
        },
        function=get_loaction_coordinate,
    ).register_function(
        name="search_nearby_pois",
        description="搜索给定坐标附近的 POI",
        parameters={
            "type": "object",
            "properties": {
                "longitude": {
                    "type": "string",
                    "description": "中心点的经度"
                },
                "latitude": {
                    "type": "string",
                    "description": "中心点的纬度"
                },
                "keyword": {
                    "type": "string",
                    "description": "目标 POI的关键字"
                }
            },
            "required": ["longitude", "latitude","keyword"]
        },
        function=search_nearby_pois,
)
result = function_calling_request.request(content="五道口附近的咖啡馆")
print("-------------\n\n",result)



