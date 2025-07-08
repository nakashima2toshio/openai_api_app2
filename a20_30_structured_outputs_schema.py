# streamlit run 01_01_1_responses_create.py --server.port=8500
#　「（Pydantic + client.responses.parse() + .output_parsed）」
#
# 08_01_01: Chain of thought
# 08_01_02: Structured data extraction
# 08_01_03: UI generation
# 08_01_04: Moderation
#
#　[Menu]----------------------------------------
# "イベント情報抽出": demo_event_extraction,
# "数学的思考ステップ": demo_math_reasoning,
# "UIコンポーネント生成": demo_ui_generation,
# "エンティティ抽出": demo_entity_extraction,
# "条件分岐スキーマ": demo_conditional_schema,
# "モデレーション＆拒否処理": demo_moderation
#　-------------------------------------------
import streamlit as st
import json
from openai import OpenAI
# --- ここから改修 ---
from openai.types.responses.response_text_config_param import ResponseTextConfigParam
from openai.types.responses.response_format_text_json_schema_config_param import (
    ResponseFormatTextJSONSchemaConfigParam
)
from a0_common_helper.helper import append_user_message
# --- ページ設定 ---
st.set_page_config(page_title="Structured Outputs Samples", page_icon="🗂️")

def create_structured_response(model, messages, schema_name, schema):
    # Structured Outputs 用 TypedDict を直接インスタンス化
    text_cfg = ResponseTextConfigParam(
        format=ResponseFormatTextJSONSchemaConfigParam(
            name=schema_name,
            schema=schema,
            type="json_schema",
            strict=True
        )
    )
    client = OpenAI()
    res = client.responses.create(
        model=model,
        input=messages,
        text=text_cfg
    )
    return json.loads(res.output_text)


# ------------------------------------
# 01_イベント情報抽出": demo_event_extraction
# ----------------------------------- """
def demo_event_extraction():
    st.header("1. イベント情報抽出デモ")
    model = st.selectbox("モデルを選択", ["o4-mini", "gpt-4o-2024-08-06", "gpt-4o-mini"])
    text = st.text_input(
        "イベント詳細を入力",
        "(例)屋台湾フェス2025 ～あつまれ！究極の屋台グルメ～ in Kawasaki Spark"
    )
    st.write("(例)屋台湾フェス2025 ～あつまれ！究極の屋台グルメ～ in Kawasaki Spark")
    if st.button("実行：イベント抽出"):
        messages = [
            {"role": "developer", "content": "Extract event details from the text."},
            {"role": "user", "content": text}
        ]
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "date": {"type": "string"},
                "participants": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["name", "date", "participants"],
            "additionalProperties": False
        }
        result = create_structured_response(
            model, messages,
            "event_extraction", schema
        )
        st.json(result)


# ------------------------------------
# 02_数学的思考ステップ": demo_math_reasoning
# ------------------------------------ """
def demo_math_reasoning():
    st.header("2. 数学的思考ステップデモ")
    model = st.selectbox(
        "モデルを選択",
        ["o4-mini", "gpt-4o-2024-08-06", "gpt-4o-mini"],
        key="math_model"
    )
    expr = st.text_input("解きたい式を入力", "8x + 7 = -23")
    if st.button("実行：思考ステップ生成"):
        messages = [
            {"role": "developer", "content": "You are a math tutor. Solve step by step."},
            {"role": "user", "content": f"how can I solve {expr}"}
        ]
        schema = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "explanation": {"type": "string"},
                            "output": {"type": "string"}
                        },
                        "required": ["explanation", "output"],
                        "additionalProperties": False
                    }
                },
                "final_answer": {"type": "string"}
            },
            "required": ["steps", "final_answer"],
            "additionalProperties": False
        }
        result = create_structured_response(
            model, messages,
            "math_reasoning", schema
        )
        st.json(result)

#  ------------------------------------
# 03_UIコンポーネント生成": demo_ui_generation,
# ------------------------------------
def demo_ui_generation():
    st.header("3. UIコンポーネント生成デモ")
    model = st.selectbox(
        "モデルを選択",
        ["o4-mini", "gpt-4o-2024-08-06", "gpt-4o-mini"],
        key="ui_model"
    )
    if st.button("実行：UI生成サンプル"):
        messages = [
            {"role": "system", "content": "Generate a recursive UI component schema."}
        ]
        schema = {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["div", "button", "header", "section", "field", "form"]},
                "label": {"type": "string"},
                "children": {"type": "array", "items": {"$ref": "#"}},
                "attributes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": "string"}
                        },
                        "required": ["name", "value"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["type", "label", "children", "attributes"],
            "additionalProperties": False
        }
        result = create_structured_response(model, messages, "ui_schema", schema)
        st.json(result)


# ------------------------------------
# 04_エンティティ抽出": demo_entity_extraction,
# ------------------------------------ """
def demo_entity_extraction():
    st.header("4. エンティティ抽出デモ")
    model = st.selectbox(
        "モデルを選択",
        ["o4-mini", "gpt-4o-2024-08-06", "gpt-4o-mini"],
        key="entity_model"
    )
    text = st.text_input(
        "抽出対象テキスト",
        "The quick brown fox jumps over the lazy dog with piercing blue eyes."
    )
    if st.button("実行：エンティティ抽出"):
        messages = [
            {"role": "system", "content": "Extract attributes, colors, and animals from the text."},
            {"role": "user", "content": text}
        ]
        schema = {
            "type": "object",
            "properties": {
                "attributes": {"type": "array", "items": {"type": "string"}},
                "colors": {"type": "array", "items": {"type": "string"}},
                "animals": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["attributes", "colors", "animals"],
            "additionalProperties": False
        }
        result = create_structured_response(model, messages, "entities", schema)
        st.json(result)

# ------------------------------------
# 05_条件分岐スキーマ": demo_conditional_schema,
# ------------------------------------
def demo_conditional_schema():
    st.header("5. 条件分岐スキーマデモ")
    model = st.selectbox(
        "モデルを選択",
        ["o4-mini", "gpt-4o-2024-08-06", "gpt-4o-mini"],
        key="cond_model"
    )
    text = st.text_input("ユーザー情報または住所を入力", "Name: Alice, Age: 30")
    if st.button("実行：条件分岐出力"):
        messages = [
            {"role": "system", "content": "Return either user info or address based on input."},
            {"role": "user", "content": text}
        ]
        schema = {
            "type": "object",
            "properties": {
                "item": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "number"}
                            },
                            "required": ["name", "age"],
                            "additionalProperties": False
                        },
                        {
                            "type": "object",
                            "properties": {
                                "number": {"type": "string"},
                                "street": {"type": "string"},
                                "city": {"type": "string"}
                            },
                            "required": ["number", "street", "city"],
                            "additionalProperties": False
                        }
                    ]
                }
            },
            "required": ["item"],
            "additionalProperties": False
        }
        result = create_structured_response(
            model, messages,
            "conditional_item", schema
        )
        st.json(result)


# ------------------------------------
# 06_モデレーション＆拒否処理": demo_moderation
# ------------------------------------ """

def demo_moderation():
    st.header("6. モデレーション＆拒否処理デモ")
    model = st.selectbox(
        "モデルを選択",
        ["o4-mini", "gpt-4o-2024-08-06", "gpt-4o-mini"],
        key="mod_model"
    )
    text = st.text_input("入力テキスト (不適切例: ...)", "Sensitive request example")
    if st.button("実行：モデレーションチェック"):
        messages = [
            {"role": "system", "content": "Check content and refuse if inappropriate."},
            {"role": "user", "content": text}
        ]
        schema = {
            "type": "object",
            "properties": {
                "refusal": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["refusal", "content"],
            "additionalProperties": False
        }
        result = create_structured_response(
            model, messages,
            "moderation", schema
        )
        st.json(result)


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
