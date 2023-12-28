import requests
import pandas as pd
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')



texts = []
with open("./source.txt") as f:
    for line in f.readlines():
        line = line.strip()
        if line == "":
            continue
        texts.append(line)



# def query(texts):
#     model_id = "xlm-roberta-base"
#     api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
#     response = requests.post(api_url, json={"inputs": texts, "options":{"wait_for_model":True}})
#     return response.json()
# output = query(texts)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

output = [ model.encode(sentence) for sentence in texts]

embeddings = pd.DataFrame(output)
print(f"len(text): {len(texts)}")
print(embeddings)

mainBoard = []
for idx in range(len(texts)):
    vec = list(embeddings.iloc[idx])
    text = texts[idx]
    mainBoard.append((idx,text,vec))

mainBoard = pd.DataFrame(mainBoard,columns=("id", "title", "vec"))
print(mainBoard)
mainBoard.to_csv("vecs.csv")