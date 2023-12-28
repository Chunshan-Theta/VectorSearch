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
REDIS_HOST =  "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = "" # default for passwordless Redis

# Connect to Redis
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD
)
assert(redis_client.ping())

data=pd.read_csv("./vecs.csv")
print(data)
# Constants
VECTOR_DIM = len(data['vec'][0]) # length of the vectors
VECTOR_NUMBER = len(data)                 # initial number of vectors
INDEX_NAME = "embeddings-index"           # name of the search index
PREFIX = "doc"                            # prefix for the document keys
DISTANCE_METRIC = "COSINE"                # distance metric for the vectors (ex. COSINE, IP, L2)
VECTOR_COLUME_NAME = "vec"
CONTENT_COLUME_NAME = "title"

# Define RediSearch fields for each of the columns in the dataset
text = TextField(name=CONTENT_COLUME_NAME)
text_embedding = VectorField(VECTOR_COLUME_NAME,
    "FLAT", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": DISTANCE_METRIC,
        "INITIAL_CAP": VECTOR_NUMBER,
    }
)
fields = [text, text_embedding]

# Check if index exists
try:
    redis_client.ft(INDEX_NAME).info()
    print("Index already exists")
except:
    # Create RediSearch Index
    redis_client.ft(INDEX_NAME).create_index(
        fields = fields,
        definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
)

def index_documents(client: redis.Redis, prefix: str, documents: pd.DataFrame):
    records = documents.to_dict("records")
    for doc in records:
        key = f"{prefix}:{str(doc['id'])}"

        # replace list of floats with byte vectors
        doc[VECTOR_COLUME_NAME] = np.array(doc[VECTOR_COLUME_NAME], dtype=np.float32).tobytes()
        _ = client.hset(key, mapping = doc)
index_documents(redis_client, PREFIX, data)
print(f"Loaded {redis_client.info()['keys']} documents in Redis search index with name: {INDEX_NAME}")

from sentence_transformers import SentenceTransformer
def embedding(text):
    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    return model.encode(text)

def search_redis(
    redis_client: redis.Redis,
    user_query: str,
    index_name: str = INDEX_NAME,
    vector_field: str = VECTOR_COLUME_NAME,
    return_fields: list = fields,
    hybrid_fields = "*",
    k: int = 20,
    print_results: bool = True,
) -> List[dict]:


    # Creates embedding vector from user query
    embedded_query = embedding(user_query)
    print(f"embedded_query: {embedded_query}")

    # Prepare the Query
    base_query = f'{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]'
    query = (
        Query(base_query)
         .return_fields(*return_fields)
         .sort_by("vector_score")
         .paging(0, k)
         .dialect(2)
    )
    
    params_dict = {"vector": np.array(embedded_query, dtype=np.float32)}

    # perform vector search
    results = redis_client.ft(index_name).search(query, params_dict)
    if print_results:
        for i, article in enumerate(results.docs):
            score = 1 - float(article.vector_score)
            print(f"{i}. {article.title} (Score: {round(score ,3) })")
    return results.docs

# For using OpenAI to generate query embedding
results = search_redis(redis_client, '出版資訊', k=10)
