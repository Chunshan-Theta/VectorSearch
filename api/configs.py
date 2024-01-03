# Create Redis Table
VECTOR_DIM = 768                          # length of the vectors
INDEX_NAME = "embeddings-index"           # name of the search index
PREFIX = "doc"                            # prefix for the document keys
DISTANCE_METRIC = "COSINE"                # distance metric for the vectors (ex. COSINE, IP, L2)
VECTOR_COLUME_NAME = "vec"
CONTENT_COLUME_NAME = "title"
TABLE_COLUME_NAME = "tableName"

# Redis Config
REDIS_HOST =  "127.0.0.1"
REDIS_PORT = 6379
REDIS_PASSWORD = "" # default for passwordless Redis