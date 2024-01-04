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
from tools import embedding, getRedis, loadDefText, textsToVecObj, newVecObj
from configs import *


# Connect to Redis
redis_client = getRedis()
# load data and convert vec to number from string
texts = loadDefText()
data = textsToVecObj(texts)


# Define RediSearch fields for each of the columns in the dataset
text = TextField(name=CONTENT_COLUME_NAME)
tableName = TextField(name=TABLE_COLUME_NAME)
text_embedding = VectorField(VECTOR_COLUME_NAME,
    "FLAT", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": DISTANCE_METRIC,
        "INITIAL_CAP": len(data),
    }
)
fields = [text, tableName, text_embedding]

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


newVecObj(redis_client, PREFIX, data)
print(f"Loaded {redis_client.info()['db0']['keys']} documents in Redis search index with name: {INDEX_NAME}")
