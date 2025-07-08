# streamlit run a20_00_responses_skeleton.py --server.port=8501
# sample Responsesサンプル
import os
import sys
import json
import base64
import glob
import requests
from pathlib import Path
import pandas as pd

# ---------------------------------------
# .env ファイルをロード
# env_path = Path(__file__).parent / ".env"
# load_dotenv(dotenv_path=env_path)
# ---------------------------------------

from openai import OpenAI
from openai.types.responses.web_search_tool_param import UserLocation

from openai.types.responses import (
    EasyInputMessageParam,      # 基本の入力メッセージ
    ResponseInputTextParam,     # 入力テキスト
    ResponseInputImageParam,    # 入力画像
    ResponseFormatTextJSONSchemaConfigParam,  # Structured output 用
    ResponseTextConfigParam,    # Structured output 用
    FunctionToolParam,          # 関数呼び出しツール
    FileSearchToolParam,        # ファイル検索ツール
    WebSearchToolParam,         # Web 検索ツール
    ComputerToolParam,          #
    Response
)

from pydantic import BaseModel, ValidationError

from helper import (
    init_page,
    init_messages,
    select_model,
    sanitize_key,
    get_default_messages,
    extract_text_from_response, append_user_message,
)

BASE_DIR = Path(__file__).resolve().parent.parent
import streamlit as st

# --- インポート直後に１度だけ実行する ---
st.set_page_config(
    page_title="ChatGPT Responses API",
    page_icon="2025-5 Nakashima"
)

# ==================================================
# sample テキスト入出力 (One Shot):responses.create
# ==================================================
def responses_create_sample(demo_name: str = "responses_create_sample"):
    init_messages(demo_name)
    st.write(f"# {demo_name}")
    model = select_model(demo_name)
    st.write("選択したモデル:", model)

    safe = sanitize_key(demo_name)
    st.write("記入例：openaiのresponses.createとresponses.parseの比較をしなさい。")
    with st.form(key=f"responses_form_{safe}"):
        user_input = st.text_area("ここにテキストを入力してください:", height=75)
        submitted = st.form_submit_button("送信")

    messages = get_default_messages()
    if submitted and user_input:
        st.write("入力内容:", user_input)
        messages.append(EasyInputMessageParam(role="user", content=user_input))
        client = OpenAI()
        res = client.responses.create(model=model, input=messages)
        for i, txt in enumerate(extract_text_from_response(res), 1):
            st.code(txt)

        with st.form(key=f"responses_next_{safe}"):
            if st.form_submit_button("次の質問"):
                st.rerun()


# ==================================================
# responses_parse_sample
# ==================================================
def responses_parse_sample(demo_name: str = "responses_parse_sample"):
    class UserInfo(BaseModel):
        name: str
        age: int
        city: str

    # ルートを object にし、その中に配列フィールドを置く
    class People(BaseModel):
        users: list[UserInfo]

    model = select_model(demo_name)
    st.write("選択したモデル:", model)
    client = OpenAI()

    safe = sanitize_key(demo_name)
    st.write(
        "(記入例："
        "私の名前は田中太郎、30歳、東京在住です。"
        "私の名前は鈴木健太、28歳、大阪在住です。"
    )
    with st.form(key=f"responses_form_{safe}"):
        user_input = st.text_area("ここにテキストを入力してください:", height=100)
        submitted = st.form_submit_button("送信")

    if submitted and user_input.strip():
        # プロンプト側で「users 配列で返して」と明示
        messages = get_default_messages()
        append_developer_text = "あなたは情報抽出アシスタントです。"
        messages.append(
            EasyInputMessageParam(
                role="developer",
                content=[
                    ResponseInputTextParam(type="input_text", text=append_developer_text),
                ]
            )
        )
        append_user_text = (
            "私の名前は田中太郎、30歳、東京在住です。"
            "私の名前は鈴木健太、28歳、大阪在住です。"
        )
        messages.append(
            EasyInputMessageParam(
                role="user",
                content=[
                    ResponseInputTextParam(type="input_text", text=append_user_text),
                ]
            )
        )

        response = client.responses.parse(
            model=model,
            input=messages,
            text_format=People
        )

        people: People = response.output_parsed
        for p in people.users:
            output = f"{p.name} / {p.age} / {p.city}"
            st.write(output)



# ==================================================
# メインルーティン
# ==================================================
def main() -> None:
    init_page("core concept")

    page_funcs = {
        "responses_create_sample" : responses_create_sample,
        "responses_parse_sample" : responses_parse_sample,
    }
    demo_name = st.sidebar.radio("デモを選択", list(page_funcs.keys()))
    st.session_state.current_demo = demo_name
    page_funcs[demo_name](demo_name)


if __name__ == "__main__":
    main()

# streamlit run a20_01_responses_parse.py --server.port 8501

