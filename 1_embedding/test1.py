import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

# 阿里百炼
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)

print(client)

import numpy as np
from numpy import dot
from numpy.linalg import norm


def cos_sim(a,b):
    '''余玄距离 -- 越大越相似'''
    return dot(a,b)/(norm(a)*norm(b))


def l2(a,b):
    '''
    欧式距离 -- 越小越相似
    '''
    x = np.asarray(a) -np.asarray(b)
    return norm(x)

def get_embeddings(texts,model='text-embedding-v1',dimensions=None):
    '''封装 OpenAi 的 Embedding 模型接口'''
    if model == 'text-embedding-v1':
        dimensions = None
    if dimensions:
        data = client.embeddings.create(
            input=texts,model=model,dimensions=dimensions
        ).data
    else:
        data = client.embeddings.create(
            input=texts,model=model
        ).data
    return [x.embedding for x in data]

test_query = ["聚客 Ai - 用科技力量，构建智能未来"]
vec = get_embeddings(test_query)[0]
print(f"total dimension : {len(vec)}")
print(f"first 10 elements : {vec[:10]}")


query = "国际争端"

documents = [
    "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
    "土耳其、芬兰、瑞典与北约代表将继续就瑞典'入北约'问题进行谈判",
    "日本支付服饰陆上自卫队射击场内发生枪击事件，3 人受伤",
    "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
    "我国首次在空间开展舱外辐射生物学暴露实验",
]

quert_vec = get_embeddings([query])[0]
docs_vecs = get_embeddings(documents)

print("Query 与自己的余弦距离: {:.2}".format(cos_sim(quert_vec,quert_vec)))
print("Query 与 document的余弦距离:")
for vec in docs_vecs:
    print(cos_sim(quert_vec,vec))

print()

print("Query 与自己的欧式距离: {:.2}".format(l2(quert_vec,quert_vec)))
print("Query 与 document的欧式距离:")
for vec in docs_vecs:
    print(l2(quert_vec,vec))








