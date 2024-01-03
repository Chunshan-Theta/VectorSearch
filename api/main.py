from typing import Union
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import initRedis
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from redis.commands.search.query import Query
import redis
from initRedis import index_documents
import uuid
app = FastAPI()
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





class TextObject(BaseModel):
    tableName: str = "tableName"
    text: str = "text"

def embedding(text):
    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    return model.encode(text)


# Connect to Redis
REDIS_HOST =  "127.0.0.1"
REDIS_PORT = 6379
REDIS_PASSWORD = "" # default for passwordless Redis
INDEX_NAME = "embeddings-index"           # name of the search index

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD
)
assert(redis_client.ping())



@app.get("/texts/")
def read_item(tableName: str = "doc", text: str = "文章來源"):
    #vectorize the query
    k=5
    vector_field = "vec"
    query_vector = embedding(text).astype(np.float32).tobytes()

    #prepare the query
    sql = f'(@tableName:{tableName})=>[KNN {k} @{vector_field} $vec_param AS vector_score]'
    # sql = f'(*)=>[KNN {k} @{vector_field} $vec_param AS vector_score]'
    q = Query(sql).sort_by('vector_score').paging(0,k).dialect(2)
    print(f"sql:{sql}")
    
    params_dict = {"vec_param": query_vector}
    #Execute the query
    results = redis_client.ft(INDEX_NAME).search(q, query_params = params_dict)
    # for i, article in enumerate(results.docs):
    #     print(f"{i}. {article.title} (Score: {round(1 - float(article.vector_score) ,3) })")
    #     print(article)
    return {"tableName": tableName, "text": text, "similar": [(article.title, 1 - float(article.vector_score)) for i, article in enumerate(results.docs)]}



@app.post("/texts/")
async def create_item(obj: TextObject):
    userText = obj.text
    randomUUID = str(uuid.uuid4())
    vec = embedding(obj.text)
    mainBoard = [[randomUUID,userText,vec]]
    index_documents(redis_client, obj.tableName, pd.DataFrame(mainBoard,columns=("id", "title", "vec")))

    return obj