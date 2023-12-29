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
import re
def embedding(text):
    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    return model.encode(text)



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

# load data and convert vec to number from string

texts = []
windowsSize = 1
with open("./source.txt") as f:
    for line in f.readlines():
        line = line.strip()
        if line == "":
            continue
        print("text:",line)
        subString = [i for i in re.split('，|;|\'|\?|\~|!|&|=|。|；',line) if len(i)>1]
        print(f"subString: {subString}")
        texts.extend(["，".join(subString[idx-windowsSize:idx+windowsSize]) for idx in range(windowsSize,len(subString)-windowsSize)])
print(f"texts: {texts}")
output = [ embedding(sentence) for sentence in texts]

embeddings = pd.DataFrame(output)
print(f"len(text): {len(texts)}")
# print(embeddings)

mainBoard = []
for idx in range(len(texts)):
    vec = list(embeddings.iloc[idx])
    text = texts[idx]
    mainBoard.append([idx,text,vec])

data = pd.DataFrame(mainBoard,columns=("id", "title", "vec"))
print(f"data:\n{data}")

# Create Redis Table
VECTOR_DIM = len(data['vec'][0]) # length of the vectors
VECTOR_NUMBER = len(data)                 # initial number of vectors
INDEX_NAME = "embeddings-index"           # name of the search index
PREFIX = "doc"                            # prefix for the document keys
DISTANCE_METRIC = "COSINE"                # distance metric for the vectors (ex. COSINE, IP, L2)
VECTOR_COLUME_NAME = "vec"
CONTENT_COLUME_NAME = "title"
print(f"VECTOR_DIM: {VECTOR_DIM}")

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
        doc[VECTOR_COLUME_NAME] = np.array(doc[VECTOR_COLUME_NAME]).astype(np.float32).tobytes()
        # print(f"size: {len(doc[VECTOR_COLUME_NAME])}")
        client.hset(key, mapping = doc)
index_documents(redis_client, PREFIX, data)
print(f"Loaded {redis_client.info()['db0']['keys']} documents in Redis search index with name: {INDEX_NAME}")
