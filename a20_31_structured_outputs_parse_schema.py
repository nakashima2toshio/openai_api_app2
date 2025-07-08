# streamlit run a20_31_structured_outputs_parse_schema.py --server.port=8501
#　[Menu]----------------------------------------
# 01 イベント情報抽出": demo_event_extraction: client.responses.parse(model=model,input=text,text_format=EventInfo,)
# 02 数学的思考ステップ": demo_math_reasoning: client.responses.parse(model=model,input=prompt,text_format=MathReasoning,)
# 03 UIコンポーネント生成": demo_ui_generation: client.responses.parse(model=model, input=prompt, text_format=UIComponent)
# 04 エンティティ抽出": demo_entity_extraction:
# 05 条件分岐スキーマ": demo_conditional_schema:
# 06 モデレーション＆拒否処理": demo_moderation:
#　-------------------------------------------
from typing import List, Union, Optional
from pydantic import BaseModel, Field

from openai import OpenAI
# from  openai.lib._tools import pydantic_function_tool
import streamlit as st

# -----------------------------------
# Responses API で利用する型 (openai-python v1)
# -----------------------------------
from openai.types.responses import (
    EasyInputMessageParam,      # 基本の入力メッセージ
    ResponseInputTextParam,     # 入力テキスト
    ResponseInputImageParam,    # 入力画像
    ResponseFormatTextJSONSchemaConfigParam,  # Structured output 用
    ResponseTextConfigParam,    # Structured output 用
    FunctionToolParam,          # 関数呼び出しツール
    FileSearchToolParam,        # ファイル検索ツール
    WebSearchToolParam,         # Web 検索ツール
    ComputerToolParam,          # AIが操作するRPA機能
    Response
)
# --------------------------------------------------
# デフォルトプロンプト
# --------------------------------------------------
def get_default_messages() -> list[EasyInputMessageParam]:
    developer_text = (
        "You are a strong developer and good at teaching software developer professionals "
        "please provide an up-to-date, informed overview of the API by function, then show "
        "cookbook programs for each, and explain the API options."
        "あなたは強力な開発者でありソフトウェア開発者の専門家に教えるのが得意です。"
        "OpenAIのAPIを機能別に最新かつ詳細に説明してください。"
        "それぞれのAPIのサンプルプログラムを示しAPIのオプションについて説明してください。"
    )
    user_text = (
        "Organize and identify the problem and list the issues. "
        "Then, provide a solution procedure for the issues you have organized and identified, "
        "and solve the problems/issues according to the solution procedures."
        "不具合、問題を特定し、整理して箇条書きで列挙・説明してください。"
        "次に、整理・特定した問題点の解決手順を示しなさい。"
        "次に、解決手順に従って問題・課題を解決してください。"
    )
    assistant_text = "OpenAIのAPIを使用するには、公式openaiライブラリが便利です。回答は日本語で"

    return [
    EasyInputMessageParam(role="developer", content=developer_text),
    EasyInputMessageParam(role="user",      content=user_text),
    EasyInputMessageParam(role="assistant", content=assistant_text),
]

# role="user"の append messageの追加
def append_message(user_input_text):
    messages = get_default_messages()
    messages.append(
        EasyInputMessageParam(role="user", content=user_input_text)
    )
    return messages

# ------------------------------------------------------
from openai.types.responses import EasyInputMessageParam           # ← ここがポイント

# ページ設定
st.set_page_config(page_title="Structured Outputs Samples", page_icon="🗂️")
# ------------------------------------------------------
# 01_イベント情報抽出: demo_event_extraction
# ------------------------------------------------------
# 1) 取り出したい構造を Pydantic で宣言 --------------------
class EventInfo(BaseModel):
    # 抽出対象：イベント情報
    name: str = Field(..., description="イベント名")
    date: str = Field(..., description="開催日")
    participants: List[str] = Field(..., description="参加者一覧")

# 2) OpenAI Responses API で構造化データを取得するユーティリティ ---
def create_structured_response(model: str, text: str) -> dict:
    # 指定モデルで text を解析し、EventInfo を dict で返す
    client = OpenAI()
    response = client.responses.parse(model=model,input=text,text_format=EventInfo,)

    # output_parsed は text_format に渡した Pydantic モデルのインスタンス
    event_info: EventInfo = response.output_parsed
    return event_info.model_dump()

# 3) streamlitのUI: 画面側ロジック ------------------------
def demo_event_extraction() -> None:
    st.header("1. イベント情報抽出デモ")

    model = st.selectbox(
        "モデルを選択",
        ["o4-mini", "gpt-4o-2024-08-06", "gpt-4o-mini"],
        index=0,
    )

    # テキスト入力
    default_text = (
        "屋台湾フェス2025 ～あつまれ！究極の屋台グルメ～ in Kawasaki Spark "
        "（5/3・5/4開催）参加者：王さん、林さん、佐藤さん"
    )
    user_text = st.text_area(
        "イベント詳細を入力: ",
        value=default_text,
        height=4 * 24  # 4行×1行の高さ(およそ24px)＝96px
    )
    st.caption(f"例）{default_text}")

    # 実行ボタン
    if st.button("実行：イベント抽出"):
        result = create_structured_response(model, user_text)

        st.subheader("抽出結果")
        st.json(result)

# --------------------------------------------------------------
# 02. 数学的思考ステップ: demo_math_reasoning
# --------------------------------------------------------------
class Step(BaseModel):
    explanation: str = Field(..., description="このステップでの説明")
    output: str = Field(..., description="このステップの計算結果")

class MathReasoning(BaseModel):
    steps: List[Step] = Field(..., description="逐次的な解法ステップ")
    final_answer: str = Field(..., description="最終解")

def parse_math_reasoning(model: str, expression: str) -> dict:
    # expression を解析し MathReasoning を返す
    prompt = (
        "You are a skilled math tutor. "
        f"Solve the equation {expression} step by step. "
        "Return the reasoning as a JSON that matches the MathReasoning schema."
    )
    client = OpenAI()
    resp = client.responses.parse(model=model,input=prompt,text_format=MathReasoning,)
    return resp.output_parsed.model_dump()

def demo_math_reasoning() -> None:
    st.header("2. 数学的思考ステップデモ")

    model = st.selectbox(
        "モデルを選択",
        ["o4-mini", "gpt-4o-2024-08-06", "gpt-4o-mini"],
        key="math_model",
    )

    expr = st.text_input("解きたい式を入力", "8x + 7 = -23")

    if st.button("実行：思考ステップ生成"):
        result = parse_math_reasoning(model, expr)
        st.subheader("思考ステップ")
        st.json(result)

# --------------------------------------------------------------
# 03. UUIコンポーネント生成: demo_ui_generation
# --------------------------------------------------------------
class UIAttribute(BaseModel):
    name: str = Field(..., description="属性名")
    value: str = Field(..., description="属性値")

class UIComponent(BaseModel):
    type: str = Field(..., description="コンポーネント種類 (div/button など)")
    label: str = Field(..., description="表示ラベル")
    children: List["UIComponent"] = Field(default_factory=list, description="子要素")
    attributes: List[UIAttribute] = Field(default_factory=list, description="属性のリスト")

    model_config = {"extra": "forbid"}  # 余計なキーを拒否

UIComponent.model_rebuild()             # 再帰型を解決

def parse_ui_component(model: str, request: str) -> dict:
    prompt = (
        "You are a front-end architect. "
        "Generate a recursive UI component tree in JSON that matches the UIComponent schema. "
        "Design the UI requested below:\n"
        f"{request}"
    )
    client = OpenAI()
    resp = client.responses.parse(model=model, input=prompt, text_format=UIComponent)
    return resp.output_parsed.model_dump()

def demo_ui_generation() -> None:
    st.header("3. UIコンポーネント生成デモ")

    model = st.selectbox("モデルを選択",
                         ["o4-mini", "gpt-4o-2024-08-06", "gpt-4o-mini"],
                         key="ui_model")

    default_req = "ログインフォーム（メールアドレスとパスワード入力欄、ログインボタン）"
    ui_request = st.text_area("生成したい UI を説明してください", value=default_req, height=72)

    if st.button("実行：UI生成"):
        st.subheader("生成された UI スキーマ")
        st.json(parse_ui_component(model, ui_request))

# --------------------------------------------------------------
# 04. Entity Extraction Demo
# --------------------------------------------------------------
class Entities(BaseModel):
    attributes: List[str]
    colors: List[str]
    animals: List[str]

def parse_entities(model: str, text: str) -> dict:
    prompt = (
        "Extract three kinds of entities from the text below:\n"
        "- attributes (形容詞・特徴)\n"
        "- colors\n"
        "- animals\n\n"
        "Return the result as JSON that matches the Entities schema.\n\n"
        f"TEXT:\n{text}"
    )
    client = OpenAI()
    resp = client.responses.parse(model=model, input=prompt, text_format=Entities)
    return resp.output_parsed.model_dump()

def demo_entity_extraction() -> None:
    st.header("4. エンティティ抽出デモ")
    model = st.selectbox("モデルを選択",
                         ["o4-mini", "gpt-4o-2024-08-06", "gpt-4o-mini"],
                         key="entity_model")
    text = st.text_input("抽出対象テキスト",
                         "The quick brown fox jumps over the lazy dog with piercing blue eyes.")
    if st.button("実行：エンティティ抽出"):
        st.json(parse_entities(model, text))
# --------------------------------------------------------------
# --------------------------- 05. Conditional Schema Demo --------------------
class UserInfo(BaseModel):
    name: str
    age: int

class Address(BaseModel):
    number: str
    street: str
    city: str

class ConditionalItem(BaseModel):
    item: Union[UserInfo, Address]
    model_config = {"extra": "forbid"}  # item 以外のキーを拒否

def parse_conditional_item(model: str, text: str) -> dict:
    prompt = (
        "You will receive either a user profile or a postal address.\n"
        "If the input represents a person, parse it into the UserInfo schema.\n"
        "If it represents an address, parse it into the Address schema.\n"
        "Wrap the result in the field 'item' and return JSON that matches the ConditionalItem schema.\n\n"
        f"INPUT:\n{text}"
    )
    client = OpenAI()
    resp = client.responses.parse(model=model, input=prompt, text_format=ConditionalItem)
    return resp.output_parsed.model_dump()

def demo_conditional_schema() -> None:
    st.header("5. 条件分岐スキーマデモ")
    model = st.selectbox("モデルを選択",
                         ["o4-mini", "gpt-4o-2024-08-06", "gpt-4o-mini"],
                         key="cond_model")
    text = st.text_input("ユーザー情報または住所を入力", "Name: Alice, Age: 30")
    if st.button("実行：条件分岐出力"):
        st.json(parse_conditional_item(model, text))

# --------------------------------------------------------------
# --------------------------- 06. Moderation & Refusal Demo ------------------
class ModerationResult(BaseModel):
    refusal: str = Field(..., description="拒否する場合は理由、問題なければ空文字")
    content: Optional[str] = Field(None, description="許可された場合の応答コンテンツ")

    model_config = {"extra": "forbid"}

def parse_moderation(model: str, text: str) -> dict:
    prompt = (
        "You are a strict content moderator. "
        "If the input violates policy (hate, sexual, violence, self-harm, etc.), "
        "set 'refusal' to a short reason and leave 'content' null. "
        "Otherwise set 'refusal' to an empty string and echo the safe content in 'content'.\n\n"
        f"INPUT:\n{text}"
    )
    client = OpenAI()
    resp = client.responses.parse(model=model, input=prompt, text_format=ModerationResult)
    return resp.output_parsed.model_dump()

def demo_moderation() -> None:
    st.header("6. モデレーション＆拒否処理デモ")
    model = st.selectbox("モデルを選択",
                         ["o4-mini", "gpt-4o-2024-08-06", "gpt-4o-mini"],
                         key="mod_model")
    text = st.text_input("入力テキスト (不適切例: ...)", "Sensitive request example")
    if st.button("実行：モデレーションチェック"):
        st.json(parse_moderation(model, text))
# --------------------------------------------------------------
# --- メイン処理: サイドバーからデモ選択 ---
def main():
    demos = {
        "01_イベント情報抽出": demo_event_extraction,
        "02_数学的思考ステップ": demo_math_reasoning,
        "03_UIコンポーネント生成": demo_ui_generation,
        "04_エンティティ抽出": demo_entity_extraction,
        "05_条件分岐スキーマ": demo_conditional_schema,
        "06_モデレーション＆拒否処理": demo_moderation
    }
    # choice = st.sidebar.selectbox("サンプルプログラムを選択", list(demos.keys()))
    choice = st.sidebar.radio(
        "サンプルプログラムを選択",
        list(demos.keys())
    )
    demos[choice]()


if __name__ == "__main__":
    main()
