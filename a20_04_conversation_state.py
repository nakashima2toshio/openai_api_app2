# a20_04_conversation_state.py
# Learn how to manage conversation state during a model interaction.
# モデルの対話中に会話の状態を管理する方法を学びます。
# ----------------------------------------------------------
# 「スキーマはクラスで宣言 → ヘルパーで自動変換」
# 今後の見通し !!!
# openai-agents 0.0.18 以降 に to_responses_tool() が追加予定とロードマップに記載あり。
# それまでは上記キャスト方式か、Pydantic から直接 FunctionToolParam を組み立てる方法が最短です。
# ----------------------------------------------------------
# ステートフルな API／要約によるウィンドウ管理／エージェント間の状態共有
# といった会話メモリ設計の全体像が把握できます。
# ----------------------------------------------------------
# 「API が会話履歴を自動で保持する (stateful)」
# 「巨大な履歴を要約してトークンを節約する」
# 「前の応答 ID を渡してスレッドを継続する」
# --------------------------------------------
# Conversation State 関連記事一覧
# 題名（日本語）	                                    題名（英語）	                                                        URL
# --------------------------------------------
# Responses APIでのWeb検索とステート管理	            Web Search and States with Responses API	                        https://cookbook.openai.com/examples/responses_api/responses_example
# Responses APIでReasoningモデルの性能を高める	        Better performance from reasoning models using the Responses API	https://cookbook.openai.com/examples/responses_api/reasoning_items
# Responses APIのFile SearchでPDFに対してRAGを行う	    Doing RAG on PDFs using File Search in the Responses API	        https://cookbook.openai.com/examples/file_search_responses
# Realtime APIでのコンテキスト要約	                    Context Summarization with Realtime API	                            https://cookbook.openai.com/examples/context_summarization_with_realtime_api
# Realtime APIでデータ集約型アプリを構築する実践ガイド	    Practical guide to data-intensive apps with the Realtime API	    https://cookbook.openai.com/examples/data-intensive-realtime-apps
# エージェントのオーケストレーション: ルーチンとハンドオフ	Orchestrating Agents: Routines and Handoffs	                        https://cookbook.openai.com/examples/orchestrating_agents
# Agents SDKで音声アシスタントを構築する	                Building a Voice Assistant with the Agents SDK	                    https://cookbook.openai.com/examples/agents_sdk/app_assistant_voice_agents
# ----------------------------------------------------------
# Responses API 系の記事（上３つ）は previous_response_id を渡すことで履歴＋内部 Reasoning チェーンを丸ごと再利用する方法を詳述。
# Realtime API 系の記事（4・5行目）は音声対話で膨らむ履歴を ConversationState クラスで保持し、トークンウィンドウ上限を超えそうなときに自動要約する実装パターンを示しています。
# Agents SDK 系（6・7行目）は複数エージェントや音声ボットで「マネージドな stateful conversation」を前提にした設計指針を紹介。
# ----------------------------------------------------------
# Web Search and States with Responses API
# --------------------------------------------
# --------------------------------------------------
# デフォルトプロンプト　（例）ソフトウェア開発用
# --------------------------------------------------
import requests
import pprint
from openai import OpenAI
from openai.types.responses.web_search_tool_param import UserLocation, WebSearchToolParam
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

from pydantic import BaseModel, ValidationError, Field

# from helper import init_page, init_messages, select_model, sanitize_key, extract_text_from_response, \
#     get_default_messages

# ----------------------------------------------------------
# sample01:
# client.responses.createでアクセスし、引き続きを
# client.responses.create,previous_response_id=response.id
# ----------------------------------------------------------
def sample01():
    client = OpenAI()

    messages = get_default_messages()
    user_text = "OpenAIのresponses.create APIの使い方を教えて。"
    messages.append(
        EasyInputMessageParam(
            role="user",
            content=[
                ResponseInputTextParam(type="input_text", text=user_text),
            ],
        )
    )
    print('(1)----------------')
    response = client.responses.create(model="gpt-4o-mini",input=messages,)
    print(response.output[0].content[0].text)
    pprint.pprint(response.model_dump())

    print('(2)----------------')
    response_two = client.responses.create(
        model="gpt-4o-mini",
        input="OpenAIのresponses.createの引数 toolsの使い方を教えて。",
        previous_response_id=response.id
    )
    print(response_two.output[0].content[0].text)
    pprint.pprint(response_two.model_dump())

# ----------------------------------------------------------
# sample02:　Webアクセスし、それをresponses.parseでstructuredにパースする。
# tools - (create) WebSearch - tools -> -> (parse) structured
# tool: WebSearchToolParam = {"type": "web_search_preview"}
# まとめ — Pydantic + クラスで “tools” を記述する最短ルート
# OpenAI Python SDK では pydantic_function_tool() ヘルパー
# （あるいは Agents SDK の function_tool デコレータ）を使うと、
# Pydantic BaseModel を渡すだけで JSON Schema を自動生成 してくれます。
# これにより、手書きの parameters 辞書や required 配列が不要になり、
# 型安全・保守性・入力検証の 3 点が一度に向上します。
# さらに Open-Meteo の Forecast API は経度・緯度を投げるだけで
# 現在気温 (temperature_2m) を返すため、
# クライアント側ではその返り値をそのまま LLM の回答に織り込めます
# ----------------------------------------------------------
def sample02():
    client = OpenAI()

    tool: WebSearchToolParam = {"type": "web_search_preview"}
    raw_response = client.responses.create(
        model="gpt-4o-mini",        # or another supported model
        input="最新のopenaiのAPIの responses.parse APIの情報は？",
        tools=[tool]
    )
    import pprint
    pprint.pprint(raw_response.model_dump())
    print('------------')

    # ---------- 3) 構造化パース例 ----------
    class APINews(BaseModel):
        title: str
        url: str

    messages = get_default_messages()
    append_user_text = "上の回答をtitleとurlだけ JSON で返して"
    messages.append(
        EasyInputMessageParam(
            role="user",
            content=[
                ResponseInputTextParam(type="input_text", text=raw_response.output_text),
                ResponseInputTextParam(type="input_text", text=append_user_text),
            ]
        )
    )

    structured = client.responses.parse(
        model="gpt-4.1",
        input=messages,
        previous_response_id=raw_response.id,
        text_format=APINews
    )
    print('----------------------------')
    # print(structured.output_parsed)
    import pprint
    pprint.pprint(structured.model_dump())

# ----------------------------------------------------------
# step3:
# 多段階の会話では、各ターンの後に推論トークンは破棄され、
# 各ステップの入力トークンと出力トークンは次のステップに投入される。
# ----------------------------
# まとめ — Pydantic + クラスで “tools” を記述する最短ルート
# OpenAI Python SDK では pydantic_function_tool() ヘルパー
# （あるいは Agents SDK の function_tool デコレータ）を使うと、
# Pydantic BaseModel を渡すだけで JSON Schema を自動生成 してくれます。
# これにより、手書きの parameters 辞書や required 配列が不要になり、
# 型安全・保守性・入力検証の 3 点が一度に向上します。
# さらに Open-Meteo の Forecast API は
# 経度・緯度を投げるだけで現在気温 (temperature_2m) を返すため、
# クライアント側ではその返り値をそのまま LLM の回答に織り込めます
# -------------------------------------
# 以下では ------------------------------
# (1) Pydantic でスキーマ宣言 →
# (2) ヘルパーで tool 化 →
# (3) client.responses.create() で呼び出し
# までを一気通貫で実装し、ポイントごとに解説します。
# ----------------------------------------------------------
def sample03():
    # 1. パラメータ定義（Pydantic）
    class WeatherParams(BaseModel):
        latitude: float = Field(..., description="緯度（10進）")
        longitude: float = Field(..., description="経度（10進）")

    # 2. 実関数
    def get_weather(latitude: float, longitude: float) -> float:
        """Open‑Meteo で現在気温(°C)を取得"""
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}&longitude={longitude}&current=temperature_2m"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data["current"]["temperature_2m"]

    # 3. JSON Schema を取得し additionalProperties=false を明示
    schema = WeatherParams.model_json_schema()
    schema["additionalProperties"] = False  # Responses API strict モードが要求

    # 4. FunctionToolParam を辞書リテラルで生成
    weather_tool: FunctionToolParam = {
        "type": "function",
        "name": "get_weather",
        "description": get_weather.__doc__,
        "parameters": schema,
        "strict": True,
    }

    # 5. OpenAI 呼び出し
    client = OpenAI()
    resp = client.responses.create(
        model="gpt-4.1",
        input="今日のパリの天気は？",
        tools=[weather_tool],
    )
    pprint.pprint(resp.model_dump())

def main():
    sample01()
    # sample02()
    # sample03()

if __name__ == '__main__':
    main()

