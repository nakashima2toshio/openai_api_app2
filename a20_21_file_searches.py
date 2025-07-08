# streamlit run a20_21_file_searches.py --server.port=8504
# !pip install openai pandas tqdm -q
import time
import pandas as pd, tempfile, os, json, textwrap
from openai import OpenAI

# ----------------------------------------------
# File Searches
# ----------------------------------------------
# 応答を生成する前に、モデルがファイル内の関連情報を検索できるようにします。
# 現時点では、一度に検索できるベクター ストアは 1 つだけなので、
# ファイル検索ツールを呼び出すときに含めることができるベクター ストアIDは、1つだけです。

def set_dataset() -> str:
    # -----
    # FAQ CSV を「Q: ... A: ...」形式のプレーンテキストに変換し、
    # 一時ファイルへ書き出してパスを返す。
    # -----
    CSV_PATH = "dataset/customer_support_faq_jp.csv"
    df = pd.read_csv(CSV_PATH)

    tmp_txt = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    with open(tmp_txt.name, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(textwrap.dedent(f"""\
                Q: {row['question']}
                A: {row['answer']}
            """))
    print("plain-text file:", tmp_txt.name)
    return tmp_txt.name     # ❷ パスを返す

def create_vector_store_and_upload(txt_path: str, upload_name: str) -> str:
    # -----
    # txt_path で指定されたテキストファイルを指定 Vector Store にアップロードし、
    # インデックス完了を待って VS_ID を返す。
    # -----
    client = OpenAI()
    vs = client.vector_stores.create(name=upload_name)
    VS_ID = vs.id

    # 一時ファイルを添付
    with open(txt_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="assistants")
    client.vector_stores.files.create(vector_store_id=VS_ID, file_id=file_obj.id)

    # ❹ インデックス完了をポーリング
    while client.vector_stores.retrieve(VS_ID).status != "completed":
        time.sleep(2)
    print("Vector Store ready:", VS_ID)
    return VS_ID

def standalone_search(vs_id, query):
    VS_ID = vs_id
    # query = "返品は何日以内？"
    client = OpenAI()
    results = client.vector_stores.search(vector_store_id=VS_ID, query=query)
    for r in results.data:
        print(f"{r.score:.3f}", r.content[0].text.strip()[:60], "...")

def file_searches():
    pass

def main():
    txt_path = set_dataset()  # ← 返り値を受け取る
    upload_name = "faq_store_jp"
    vs_id = create_vector_store_and_upload(txt_path, upload_name)
    query = "返品は何日以内？"
    standalone_search(vs_id, query)


if __name__ == "__main__":
    main()
