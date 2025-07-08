# streamlit run a20_01_responses_parse.py --server.port 8501
# port Check: lsof -i :5678
# 推論が有効なモデルを使用してResponses APIに API リクエスト
# OpenAI API: https://platform.openai.com/docs/api-reference/introduction
# Streamlit API: https://docs.streamlit.io/develop/api-reference
# ----------------------------------------
# [Menu] OpenAI APIの概要
# 01_01  Responsesサンプル
# 01_011 Responsesサンプル
# 01_02  画像入力(URL)
# 01_021 画像入力(base64)
# 01_03  構造化出力-responses.create-API
# 01_031 構造化出力-responses.parse-API
# 01_04  関数 calling
# 01_05  会話状態
# 01_06  ツール:FileSearch, WebSearch
# 01_061 File Search
# 01_062 Web Search
# 01_07  Computer Use Tool Param
# ----------------------------------------
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import json
import base64
import glob
import requests
from pathlib import Path
import pandas as pd

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

BASE_DIR = Path(__file__).resolve().parent.parent
THIS_DIR = Path(__file__).resolve().parent
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')

from helper import (
    init_page,
    init_messages,
    select_model,
    sanitize_key,
    get_default_messages,
    extract_text_from_response, append_user_message,
)

import streamlit as st

# --- インポート直後に１度だけ実行する ---
st.set_page_config(
    page_title="ChatGPT Responses API",
    page_icon="2025-5 Nakashima"
)

# サンプル画像 URL
image_path_sample = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/"
    "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-"
    "Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
)

# ==================================================
# 01_00 テキスト入出力 (One Shot):responses.create
# ==================================================
def responses_parse_basic(demo_name: str = "01_00_responses_parse_basic"):
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
# 01_01 テキスト入出力 (One Shot):responses.create
# ==================================================
def responses_sample(demo_name: str = "01_01_responses_One_Shot"):
    init_messages(demo_name)
    st.write(f"# {demo_name}")
    model = select_model(demo_name)
    st.write("選択したモデル:", model)

    safe = sanitize_key(demo_name)
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
# 01_011 テキスト入出力 + history: responses.create
# ==================================================
def responses_memory_sample(demo_name: str = "01_011_responses_memory"):
    init_messages(demo_name)
    st.write(f"# {demo_name}")
    model = select_model(demo_name)
    st.write("選択したモデル:", model)

    messages = get_default_messages()
    if "responses_memory_history" not in st.session_state:
        st.session_state.responses_memory_history = messages

    # 履歴表示
    for msg in st.session_state.responses_memory_history:
        role = msg["role"]
        if role == "user":
            st.markdown(f"**User:** {msg['content']}")
        elif role == "assistant":
            st.markdown(f"<span style='color:green'><b>Assistant:</b> {msg['content']}</span>", unsafe_allow_html=True)
        elif role == "developer":
            st.markdown(f"<span style='color:gray'><i>System:</i> {msg['content']}</span>", unsafe_allow_html=True)

    safe = sanitize_key(demo_name)
    with st.form(key=f"qam_form_{safe}"):
        user_input = st.text_area("ここにテキストを入力してください:", height=75, key=f"memory_input_{safe}")
        submitted = st.form_submit_button("送信")

    if submitted and user_input:
        st.session_state.responses_memory_history.append(EasyInputMessageParam(role="user", content=user_input))
        client = OpenAI()
        res = client.responses.create(model=model, input=st.session_state.responses_memory_history)
        for txt in extract_text_from_response(res):
            st.session_state.responses_memory_history.append(EasyInputMessageParam(role="assistant", content=txt))
        st.rerun()

    if st.button("会話履歴クリア", key=f"memory_clear_{safe}"):
        st.session_state.responses_memory_history = messages
        st.rerun()


# ==================================================
# 01_02 画像入力 (URL):responses.create , テキスト出力
# ==================================================
def responses_01_02_passing_url(demo_name: str = "01_02_Image_URL"):
    init_messages(demo_name)
    model = select_model(demo_name)
    st.write("選択したモデル:", model)

    safe = sanitize_key(demo_name)
    image_url = st.text_input("画像URLを入力してください", value=image_path_sample, key=f"img_url_{safe}")
    question_text = "このイメージを説明しなさい。"

    with st.form(key=f"responses_img_form_{safe}"):
        submitted = st.form_submit_button("画像で質問")

    messages = get_default_messages()
    if submitted:
        client = OpenAI()
        messages.append(
            EasyInputMessageParam(
                role="user",
                content=[
                    ResponseInputTextParam(type="input_text", text=question_text),
                    ResponseInputImageParam(type="input_image", image_url=image_url, detail="auto"),
                ],
            )
        )
        res = client.responses.create(model=model, input=messages)
        st.write(getattr(res, "output_text", str(res)))


# ==================================================
# 01_021 画像入力 (base64):responses.create
# ==================================================
def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def responses_01_021_base64_image(demo_name: str = "01_021_Image_Base64"):
    init_messages(demo_name)
    st.write(f"# {demo_name}")
    model = select_model(demo_name)
    st.write("選択したモデル:", model)

    image_dir = "images/"
    safe = sanitize_key(demo_name)
    files = sorted(
        glob.glob(f"{image_dir}/*.png") + glob.glob(f"{image_dir}/*.jpg") +
        glob.glob(f"{image_dir}/*.jpeg") + glob.glob(f"{image_dir}/*.webp") +
        glob.glob(f"{image_dir}/*.gif")
    )
    if not files:
        st.warning(f"画像ディレクトリ {image_dir} に画像ファイルがありません")
        return

    file_path = st.selectbox("画像ファイルを選択してください", files, key=f"img_select_{safe}")

    with st.form(key=f"img_b64_form_{safe}"):
        submitted = st.form_submit_button("選択画像で実行")

    if submitted:
        b64 = encode_image(file_path)
        st.image(file_path, caption="選択画像", width=320)
        messages = get_default_messages()
        messages.append(
            EasyInputMessageParam(
                role="user",
                content=[
                    ResponseInputTextParam(type="input_text", text="what's in this image?"),
                    ResponseInputImageParam(type="input_image", image_url=f"data:image/jpeg;base64,{b64}",
                                            detail="auto"),
                ],
            ),
        )
        res = OpenAI().responses.create(model=model, input=messages)
        st.subheader("出力テキスト:")
        st.write(getattr(res, "output_text", str(res)))


# ==================================================
# 01_03 構造化出力 (JSON Schema):responses.create
# ==================================================

# ------------- Pydantic モデル -------------
class Event(BaseModel):
    name: str
    date: str
    participants: list[str]


# ------------- 共通ユーティリティ ----------------
def responses_01_03_structured_output(demo_name: str = "01_03_Structured_Output") -> None:
    # Structured Outputs デモ (parse_raw 廃止対応)

    init_messages(demo_name)
    st.header("1. structured_output: イベント情報抽出デモ")
    safe = sanitize_key(demo_name)

    # モデル選択
    model = st.selectbox(
        "モデルを選択",
        ["gpt-4.1", "o4-mini", "gpt-4o-2024-08-06", "gpt-4o-mini"],
        key=f"struct_model_{safe}",
    )

    # 入力テキスト
    text = st.text_input(
        "イベント詳細を入力",
        "(例)台湾フェス2025 ～あつまれ！究極の台湾グルメ～ in Kawasaki Spark",
        key=f"struct_input_{safe}",
    )
    st.write("(例)台湾フェス2025 ～あつまれ！究極の台湾グルメ～ in Kawasaki Spark")

    if st.button("実行：イベント抽出", key=f"struct_btn_{safe}"):
        # 1. JSON Schema
        schema = {
            "type"                : "object",
            "properties"          : {
                "name"        : {"type": "string"},
                "date"        : {"type": "string"},
                "participants": {"type": "array", "items": {"type": "string"}},
            },
            "required"            : ["name", "date", "participants"],
            "additionalProperties": False,
        }

        # 2. 入力メッセージ
        messages: list[EasyInputMessageParam] = [
            EasyInputMessageParam(role="developer", content="Extract event details from the text."),
            EasyInputMessageParam(role="user", content=[ResponseInputTextParam(type="input_text", text=text)]),
        ]

        # 3. Structured Output 指定 (最新 SDK は text=ResponseTextConfigParam)
        text_cfg = ResponseTextConfigParam(
            format=ResponseFormatTextJSONSchemaConfigParam(
                name="event_extraction",
                type="json_schema",
                schema=schema,
                strict=True,
            )
        )

        # 4. API 呼び出し
        client = OpenAI()
        res = client.responses.create(model=model, input=messages, text=text_cfg)

        # 5. Pydantic でバリデート
        try:
            event: Event = Event.model_validate_json(res.output_text)
            st.subheader("抽出結果 (Pydantic)")
            st.json(event.model_dump())
            st.code(repr(event), language="python")
        except (ValidationError, json.JSONDecodeError) as err:
            st.error("出力のパースに失敗しました。モデル出力を確認してください。")
            st.exception(err)


# ==================================================
# 01_031 構造化出力 (JSON Schema):responses.parse
# ==================================================
# --- 1. Pydantic モデル ---
class Event2(BaseModel):
    name: str
    date: str
    participants: list[str]


# --- 2. コア関数 ---
def responses_01_031_structured_output(demo_name: str = "01_03_Structured_Output"):
    init_messages(demo_name)
    safe = sanitize_key(demo_name)
    st.header("1. structured_output: イベント情報抽出デモ")

    model = st.selectbox(
        "モデルを選択",
        ["o4-mini", "gpt-4o-2024-08-06", "gpt-4o-mini"],
        key=f"struct_model_{safe}",
    )
    text = st.text_input(
        "イベント詳細を入力",
        "(例)台湾フェス2025 ～あつまれ！究極の台湾グルメ～ in Kawasaki Spark",
        key=f"struct_input_{safe}",
    )

    if st.button("実行：イベント抽出", key=f"struct_btn_{safe}"):
        # メッセージを型付きで用意
        messages: list[EasyInputMessageParam] = [
            EasyInputMessageParam(
                role="developer",
                content="Extract event details from the text."
            ),
            EasyInputMessageParam(
                role="user",
                content=[ResponseInputTextParam(type="input_text", text=text)],
            ),
        ]

        # 3. parse helper を使用
        client = OpenAI()
        res = client.responses.parse(
            model=model,
            input=messages,
            text_format=Event2,  # ← ここがポイント
        )

        # 4. 返却は自動で Event2 に！
        try:
            event: Event2 = res.output_parsed  # SDK が生成
            st.subheader("抽出結果 (Pydantic)")
            st.json(event.model_dump())
            st.code(repr(event), language="python")
        except (ValidationError, AttributeError) as ve:
            st.error("Pydantic パースに失敗しました。")
            st.exception(ve)


# ==================================================
# 01_04 関数呼び出し (OpenWeatherMap): Function calling by use json-format
# ==================================================
function_tool_param: FunctionToolParam = {
    "name"       : "get_current_weather",
    "description": "指定都市の現在の天気を返す",
    "parameters" : {
        "type"      : "object",
        "properties": {
            "location": {"type": "string"},
            "unit"    : {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required"  : ["location"],
    },
    "strict"     : True,
    "type"       : "function",
}


def load_japanese_cities(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    jp = df[df["country"] == "Japan"][["name", "lat", "lon"]].drop_duplicates()
    return jp.sort_values("name").reset_index(drop=True)


def select_city(df_jp: pd.DataFrame, demo_name: str = ""):
    safe = sanitize_key(demo_name)
    city = st.selectbox("都市を選択してください", df_jp["name"].tolist(), key=f"city_{safe}")
    row = df_jp[df_jp["name"] == city].iloc[0]
    return city, row["lat"], row["lon"]

def get_current_weather_by_coords(lat: float, lon: float, unit: str = "metric"):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENWEATHER_API_KEY が設定されていません。")
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units={unit}"
    res = requests.get(url)
    res.raise_for_status()
    data = res.json()
    return {
        "city"       : data["name"],
        "temperature": data["main"]["temp"],
        "description": data["weather"][0]["description"],
        "coord"      : data["coord"],
    }


def get_weekly_forecast(lat: float, lon: float, unit: str = "metric"):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENWEATHER_API_KEY が設定されていません。")
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units={unit}&appid={api_key}"
    res = requests.get(url)
    res.raise_for_status()
    data = res.json()
    daily: dict[str, dict[str, list[float] | str]] = {}
    for item in data["list"]:
        date = item["dt_txt"].split(" ")[0]
        temp = item["main"]["temp"]
        weather = item["weather"][0]["description"]
        daily.setdefault(date, {"temps": [], "weather": weather})["temps"].append(temp)
    return [
        {"date": d, "temp_avg": round(sum(v["temps"]) / len(v["temps"]), 1), "weather": v["weather"]}
        for d, v in daily.items()
    ]


def responses_01_04_function_calling(demo_name: str = "01_04_Function_Calling"):
    init_messages(demo_name)
    model = select_model(demo_name)
    st.write("選択したモデル:", model)

    df_jp = load_japanese_cities("data/cities_list.csv")
    city, lat, lon = select_city(df_jp, demo_name)

    today = get_current_weather_by_coords(lat, lon)
    st.write("----- 本日の天気 -----")
    st.write(f"都市 : {today['city']}")
    st.write(f"気温 : {today['temperature']}℃")
    st.write(f"説明 : {today['description']}")

    st.write("----- 5日間予報 （3時間毎を日別平均） -----")
    for day in get_weekly_forecast(lat, lon):
        st.write(f"{day['date']} : {day['temp_avg']}℃, {day['weather']}")


# --------------------------------------------------
# 01_05　会話状態
# --------------------------------------------------
def responses_01_05_conversation(demo_name: str = "01_05_Conversation"):
    init_messages(demo_name)
    model = select_model(demo_name)
    st.write("選択したモデル:", model)


# --------------------------------------------------
# 01_06 Built-in Tools (FileSearch / WebSearch)
# --------------------------------------------------
def responses_01_06_tools_file_search(demo_name: str = "01_06_Extend_Model") -> None:
    # FileSearch / WebSearch プレビュー デモ（改訂版）
    init_messages(demo_name)

    model = select_model(demo_name)
    st.write("選択したモデル:", model)

    # --- UI -----------------------------------------------------------
    tool_choice = st.selectbox("ツール選択", ["file_search", "web_search_preview"], key=f"tool_{demo_name}")
    query = st.text_input("クエリを入力", "", key=f"query_{demo_name}")

    # FileSearch 用追加パラメータ
    vector_store_id: str | None = None
    max_results: int = 5
    if tool_choice == "file_search":
        vector_store_id = st.text_input("vector_store_id", "", key=f"vs_{demo_name}")
        max_results = st.number_input("最大取得数", 1, 20, 5, key=f"max_{demo_name}")

    # --- 実行 ---------------------------------------------------------
    if st.button("送信（Tools）", key=f"btn_{demo_name}"):
        client = OpenAI()

        if tool_choice == "file_search":
            # 1. FileSearchToolParam を生成
            fs_tool = FileSearchToolParam(
                type="file_search",
                vector_store_ids=[vector_store_id] if vector_store_id else [],
                max_num_results=int(max_results),
            )

            # 2. Responses API 呼び出し（検索結果を同時返却）
            resp = client.responses.create(
                model=model,
                tools=[fs_tool],
                input=query,
                include=["file_search_call.results"],  # 🔑 ここが新規
            )

            # 3. 結果表示
            st.subheader("モデル回答")
            st.write(getattr(resp, "output_text", str(resp)))

            st.subheader("FileSearch 結果")
            if resp.file_search_call and resp.file_search_call.results:
                st.json(resp.file_search_call.results)
            else:
                st.info("検索結果が返されませんでした。vector_store_id とクエリを確認してください。")

        else:  # --- web_search_preview ----------------------------------
            ws_tool = WebSearchToolParam(
                type="web_search_preview",
                search_context_size="medium",
            )
            resp = client.responses.create(model=model, tools=[ws_tool], input=query)
            st.subheader("モデル回答")
            st.write(getattr(resp, "output_text", str(resp)))


# --------------------------------------------------
# 01_061 Built-in Tools (FileSearch)
# 想定シナリオ
# ------------
# FileSearch
# 自前でアップロードしたファイル（PDF／MD／DOCX など）を対象に、
# ベクトル検索 + キーワード検索を行い、モデル回答の根拠となるテキストや引用を取り出す
# 社内マニュアルや研究論文の Q&A、FAQ ボット、RAG 構築
# ------------
# 2. FileSearch の機能
# ・ベクトルストア連携: 事前に vector_store を作成し、files.upload() で文書を追加→自動埋め込み。
# ・意味検索 + キーワード検索: GPT‑4o がクエリを生成→ストアを検索→最適なチャンクを取得。
# ・ファイル引用: モデル出力に file_citation アノテーションを付与し、根拠箇所を示す。
# ・検索結果取得: include=["file_search_call.results"] を指定すると、
# 　検索結果メタデータ（スコア・抜粋テキスト）を JSON で受け取れる。
# ・メタデータフィルタ: アップロード時に付与した属性（例: {"type":"pdf"}）で絞り込み可能。
# --------------------------------------------------
def responses_01_061_filesearch(demo_name: str = "01_061_filesearch") -> None:
    init_messages(demo_name)
    model = select_model(demo_name)
    vector_store_id = 'vs_68345a403a548191817b3da8404e2d82'

    fs_tool = FileSearchToolParam(
        type="file_search",
        vector_store_ids=[vector_store_id],
        max_num_results=20
    )
    client = OpenAI()
    resp = client.responses.create(
        model=model,
        tools=[fs_tool],
        input="請求書の支払い期限は？",
        include=["file_search_call.results"]
    )
    st.write(resp.output_text)


# --------------------------------------------------
# WebSearch
# インターネット上の最新情報を取得し、モデルの知識をリアルタイムで拡張する
# ニュースの要約、最新統計の取得、競合リサーチ、株価・スポーツ速報
# --------------------------------------------------
# 3. WebSearch の機能
# 外部検索エンジン: モデルが裏側で Bing / DuckDuckGo API などを使用してクエリを発行。
# 検索文脈サイズ: search_context_size（"small" / "medium" / "large"）で取得記事数と抜粋長を調整。
# 要約 + 引用: モデルは取得したページの要約と URL 引用を含めた回答を生成。
# --------------------------------------------------
def responses_01_062_websearch(demo_name: str = "01_062_websearch") -> None:
    init_messages(demo_name)
    model = select_model(demo_name)

    user_location = UserLocation(
        type="approximate",
        country="JP",
        city="Tokyo",
        region="Tokyo"
    )

    ws_tool = WebSearchToolParam(
        type="web_search_preview",
        user_location=user_location,
        search_context_size="high"  # "low", "medium", または "high"
    )

    client = OpenAI()
    # ★ モデルは gpt-4o または gpt-4o-mini にしてください
    response = client.responses.create(
        model=model,  # ここを修正
        tools=[ws_tool],
        input="週末の東京の天気とおすすめの屋内アクティビティは？"
    )
    st.write(response.output_text)

# --------------------------------------------------
# computer_use_tool_param
# --------------------------------------------------
def responses_01_07_computer_use_tool_param():
    client = OpenAI()

    # --自動実行の指示-------------------------------------------
    cu_tool = ComputerToolParam(
        type="computer_use_preview",  # fixed literal
        display_width=1280,
        display_height=800,
        environment="browser",  # "browser", "mac", "windows", or "ubuntu"
    )

    # 必須: EasyInputMessageParam を利用する（OpenAI API v1 仕様）
    messages = [
        EasyInputMessageParam(
            role="user",
            content=[
                ResponseInputTextParam(
                    type="input_text",
                    text="ブラウザで https://news.ycombinator.com を開いて、トップ記事のタイトルをコピーしてメモ帳に貼り付けて"
                )
            ]
        )
    ]

    response = client.responses.create(
        model="computer-use-preview",  # dedicated hosted‑tool model
        tools=[cu_tool],
        input=messages,
        truncation="auto",  # MUST be "auto" for this model
        stream=False,  # optional
        include=["computer_call_output.output.image_url"]  # block name: ComputerUseはこれ
    )
    import pprint
    pprint.pprint(response)
    # --------------------------------------------
    # 自動実行
    # --------------------------------------------
    for output in response.output:
        if hasattr(output, 'type') and output.type == 'computer_call':
            # 画像取得や操作内容表示
            if hasattr(output, 'action'):
                print('Action:', output.action)
            if hasattr(output, 'image_url'):
                print('Image URL:', output.image_url)


# responses_01_07_computer_use_tool_param()

# ==================================================
# メインルーティン
# ==================================================
def main() -> None:
    init_page("core concept")

    page_funcs = {
        "01_00 responses.parseの基本"        : responses_parse_basic,
        "01_01  Responsesサンプル(One Shot)" : responses_sample,
        "01_011 Responsesサンプル(History)"   : responses_memory_sample,
        "01_02  画像入力(URL)"               : responses_01_02_passing_url,
        "01_021 画像入力(base64)"            : responses_01_021_base64_image,
        "01_03  構造化出力-responses"        : responses_01_03_structured_output,
        "01_031 構造化出力-parse"            : responses_01_031_structured_output,
        "01_04  関数 calling"                : responses_01_04_function_calling,
        "01_05  会話状態"                    : responses_01_05_conversation,
        "01_06  ツール:FileSearch, WebSearch": responses_01_06_tools_file_search,
        "01_061 File Search"                 : responses_01_061_filesearch,
        "01_062 Web Search"                  : responses_01_062_websearch,
        "01_07  Computer Use Tool Param"     : responses_01_07_computer_use_tool_param,
    }
    demo_name = st.sidebar.radio("デモを選択", list(page_funcs.keys()))
    st.session_state.current_demo = demo_name
    page_funcs[demo_name](demo_name)


if __name__ == "__main__":
    main()

# streamlit run a20_01_responses_parse.py --server.port 8501
