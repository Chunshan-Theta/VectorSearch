from typing import Union
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
import initRedis
import uuid
from tools import embedding, textToVec, getQuery, getRedis, newVecObj, textsToVecObj
from configs import *
from fastapi import FastAPI, Request


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)





class TextObject(BaseModel):
    tableName: str = "doc"
    text: str = "文章來源"




redis_client = getRedis()


@app.get("/texts/")
def read_item(request: Request, tableName: str = "doc", text: str = "文章來源"):
    client_IP = request.client.host

    #vectorize the query
    k=5
    vector_field = "vec"
    query_vector = textToVec(text)

    #prepare the query
    sql = f'(@tableName:{tableName})=>[KNN {k} @{vector_field} $vec_param AS vector_score]'
    # sql = f'(*)=>[KNN {k} @{vector_field} $vec_param AS vector_score]'
    q = getQuery(sql).sort_by('vector_score').paging(0,k).dialect(2)
    
    params_dict = {"vec_param": query_vector}
    #Execute the query
    results = redis_client.ft(INDEX_NAME).search(q, query_params = params_dict)
    # for i, article in enumerate(results.docs):
    #     print(f"{i}. {article.title} (Score: {round(1 - float(article.vector_score) ,3) })")
    #     print(article)
    return {
        "client_IP": client_IP,
        "tableName": tableName, 
        "text": text, 
        "similar": [(article.title, 1 - float(article.vector_score)) for i, article in enumerate(results.docs)]
    }



@app.post("/texts/")
async def create_item(request: Request, obj: TextObject):
    client_IP = request.client.host
    userText = obj.text
    randomUUID = str(uuid.uuid4())
    newVecObj(redis_client, obj.tableName, textsToVecObj([obj.text],[randomUUID]))

    return {
        "obj": obj,
        "uuid": randomUUID,
        "client_IP": client_IP
    }