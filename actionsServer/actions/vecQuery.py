import redis
from redis.commands.search.indexDefinition import (
    IndexDefinition,
    IndexType
)
from redis.commands.search.query import Query
from redis.commands.search.field import (
    TextField,
    VectorField
)
import pandas as pd
import numpy as np
import requests
from typing import List
from sentence_transformers import SentenceTransformer

def embedding(text):
    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    return model.encode(text)



REDIS_HOST =  "redis"
REDIS_PORT = 6379
REDIS_PASSWORD = "" # default for passwordless Redis

# Connect to Redis
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD
)
assert(redis_client.ping())
# Info of Redis Table
INDEX_NAME = "embeddings-index"           # name of the search index
PREFIX = "doc"                            # prefix for the document keys
DISTANCE_METRIC = "COSINE"                # distance metric for the vectors (ex. COSINE, IP, L2)
VECTOR_COLUME_NAME = "vec"
CONTENT_COLUME_NAME = "title"
FIELDS = "text,vec"

def search_redis(
    redis_client: redis.Redis,
    user_query: str,
    index_name: str = INDEX_NAME,
    vector_field: str = VECTOR_COLUME_NAME,
    fields: list = FIELDS,
    k: int = 20,
    print_results: bool = True,
) -> List[dict]:



    #vectorize the query
    query_vector = embedding(user_query).astype(np.float32).tobytes()

    #prepare the query
    q = Query(f'*=>[KNN {k} @{vector_field} $vec_param AS vector_score]').sort_by('vector_score').paging(0,k).dialect(2)
    params_dict = {"vec_param": query_vector}


    #Execute the query
    results = redis_client.ft(index_name).search(q, query_params = params_dict)


    # if print_results:
    #     for i, article in enumerate(results.docs):
    #         score = 1 - float(article.vector_score)
    #         print(f"{i}. {article.title} (Score: {round(score ,3) })")

    return [
        (article.title, 1 - float(article.vector_score))
        for i, article in enumerate(results.docs)
    ]

# For using OpenAI to generate query embedding
# text = "台灣需要自己訓練模型嗎"
# print(f"\t\t\t{text}")
# print(search_redis(redis_client, text, k=10))

# text = "文章來源"
# print(f"\t\t\t{text}")
# print(search_redis(redis_client, text, k=10))

# text = "訓練模型如何服務公部門"
# print(f"\t\t\t{text}")
# print(search_redis(redis_client, text, k=10))
