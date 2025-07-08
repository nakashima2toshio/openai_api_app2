#　utils/get_embedding.py
from openai import OpenAI

client = OpenAI()
def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    # エンコーディング形式は？
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    # レスポンスから埋め込みベクトル部分を抽出
    embedding_vector = response["data"][0]["embedding"]
    return embedding_vector

