# common pkg
import requests
from typing import List

# data tools pkg
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re

# redis tool
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
from configs import *



##
def embedding(text):
    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    return model.encode(text)

def textToVec(text):
    return embedding(text).astype(np.float32).tobytes()

def getRedis():
    # Connect to Redis
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD
    )
    assert(redis_client.ping())
    return redis_client

def getQuery(sql):
    return Query(sql)

def getSearch(sql):
    return Query(sql).sort_by('vector_score').paging(0,k).dialect(2)

def loadDefText():
    texts = []
    windowsSize = 1
    with open("./source.txt") as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            # print("text:",line)
            subString = [i for i in re.split('，|;|\'|\?|\~|!|&|=|。|；',line) if len(i)>1]
            # print(f"subString: {subString}")
            texts.extend(["，".join(subString[idx-windowsSize:idx+windowsSize]) for idx in range(windowsSize,len(subString)-windowsSize)])
    return texts
def textsToVecObj(texts: List[str], idxs: List[str] = None):
    mainBoard = []
    output = [ embedding(sentence) for sentence in texts]
    embeddings = pd.DataFrame(output)
    for idx in range(len(texts)):
        vec = list(embeddings.iloc[idx])
        text = texts[idx]
        idLabel = idx if idxs is None else idxs[idx]
        mainBoard.append([idLabel,text,vec])

    return pd.DataFrame(mainBoard,columns=("id", "title", VECTOR_COLUME_NAME))
    

def newVecObj(client: redis.Redis, prefix: str, documents: pd.DataFrame, ttl: int = None):
    records = documents.to_dict("records")
    for doc in records:
        key = f"{prefix}:{str(doc['id'])}"

        # replace list of floats with byte vectors
        doc[VECTOR_COLUME_NAME] = np.array(doc[VECTOR_COLUME_NAME]).astype(np.float32).tobytes()
        doc['tableName'] = prefix
        # print(f"size: {len(doc[VECTOR_COLUME_NAME])}")
        # print(f"doc:\n{doc}")
        client.hset(key, mapping = doc)
        if ttl is not None:
            client.expire(key, ttl) #sec