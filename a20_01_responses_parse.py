# streamlit run a20_01_responses_parse.py --server.port 8501
# port Check: lsof -i :5678
# 推論が有効なモデルを使用してResponses APIに API リクエスト
# OpenAI API: https://platform.openai.com/docs/api-reference/introduction
# Streamlit API: https://docs.streamlit.io/develop/api-reference
# ----------------------------------------
# [Menu] OpenAI APIの概要　（改修前）
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
# streamlit run a20_01_responses_parse.py --server.port=8501
# 改修版: 新しいヘルパーモジュールを使用したResponses API包括サンプル
# ==================================================
# OpenAI Responses APIの様々な機能をデモンストレーション
# - 基本的なAPI呼び出し
# - 画像入力（URL/Base64）
# - 構造化出力
# - 関数呼び出し
# - ツール（FileSearch/WebSearch）
# - Computer Use
# ==================================================

import os
import sys
import json
import base64
import glob
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field

import streamlit as st
from openai import OpenAI
from openai.types.responses.web_search_tool_param import UserLocation

from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputTextParam,
    ResponseInputImageParam,
    ResponseFormatTextJSONSchemaConfigParam,
    ResponseTextConfigParam,
    FunctionToolParam,
    FileSearchToolParam,
    WebSearchToolParam,
    ComputerToolParam,
    Response
)

# 改修されたヘルパーモジュールをインポート
try:
    from helper_st import (
        UIHelper, MessageManagerUI, ResponseProcessorUI,
        SessionStateManager, error_handler_ui, timer_ui,
        init_page, select_model
    )
    from helper_api import (
        config, logger, TokenManager, OpenAIClient,
        EasyInputMessageParam, ResponseInputTextParam
    )
except ImportError as e:
    st.error(f"ヘルパーモジュールのインポートに失敗しました: {e}")
    st.stop()


# ==================================================
# Pydantic モデル定義
# ==================================================
class UserInfo(BaseModel):
    name: str = Field(..., description="ユーザー名")
    age: int = Field(..., ge=0, le=150, description="年齢")
    city: str = Field(..., description="居住都市")


class People(BaseModel):
    users: List[UserInfo] = Field(..., description="ユーザー一覧")
    total_count: int = Field(..., description="総人数")


class Event(BaseModel):
    name: str = Field(..., description="イベント名")
    date: str = Field(..., description="開催日")
    participants: List[str] = Field(..., description="参加者一覧")


# ==================================================
# 情報パネル管理
# ==================================================
class InfoPanelManager:
    """左ペインの情報パネル管理"""

    @staticmethod
    def show_api_info():
        """API情報パネル"""
        with st.sidebar.expander("🔧 API情報", expanded=True):
            st.write("**利用可能な機能**")
            api_features = [
                "responses.create - 基本対話",
                "responses.parse - 構造化出力",
                "画像入力 (URL/Base64)",
                "関数呼び出し",
                "FileSearch ツール",
                "WebSearch ツール",
                "Computer Use ツール"
            ]
            for feature in api_features:
                st.write(feature)

    @staticmethod
    def show_model_capabilities(selected_model: str):
        """モデル能力情報"""
        with st.sidebar.expander("モデル能力", expanded=False):
            limits = TokenManager.get_model_limits(selected_model)

            # モデルカテゴリ判定
            categories = config.get("models.categories", {})
            model_category = "standard"
            for category, models in categories.items():
                if selected_model in models:
                    model_category = category
                    break

            # カテゴリ別特徴表示
            if "reasoning" in model_category:
                st.info("推論モデル - 複雑な問題解決に最適")
                st.write("- 段階的思考")
                st.write("- 論理的推論")
                st.write("- 問題分解")
            elif "audio" in selected_model:
                st.info("🎵 音声モデル - 音声入出力対応")
                st.write("- 音声認識")
                st.write("- 音声合成")
                st.write("- リアルタイム対話")
            elif "vision" in selected_model or "gpt-4o" in selected_model:
                st.info("マルチモーダルモデル - 画像理解対応")
                st.write("- 画像解析")
                st.write("- 文書読取")
                st.write("- 図表理解")
            else:
                st.info("標準モデル - 汎用的な対話")

            # トークン制限
            col1, col2 = st.columns(2)
            with col1:
                st.metric("最大入力", f"{limits['max_tokens']:,}")
            with col2:
                st.metric("最大出力", f"{limits['max_output']:,}")

    @staticmethod
    def show_demo_guide():
        """デモガイド"""
        with st.sidebar.expander("📚 デモガイド", expanded=False):
            st.write("**基本機能**")
            st.write("- Parse Basic: 構造化データ抽出")
            st.write("- Create: 基本対話")
            st.write("- Memory: 履歴付き対話")

            st.write("**画像機能**")
            st.write("- Image URL: URL指定画像解析")
            st.write("- Image Base64: ファイル画像解析")

            st.write("**高度な機能**")
            st.write("- Structured: JSON構造化出力")
            st.write("- Function: 外部関数呼び出し")
            st.write("- Tools: 検索・操作ツール")


# ==================================================
# ユーティリティ関数
# ==================================================
def encode_image(path: str) -> str:
    """画像ファイルをBase64エンコード"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_weather_function_tool() -> FunctionToolParam:
    """天気取得関数ツールの定義"""
    return {
        "name"       : "get_current_weather",
        "description": "指定都市の現在の天気を返す",
        "parameters" : {
            "type"      : "object",
            "properties": {
                "location": {"type": "string", "description": "都市名"},
                "unit"    : {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "温度単位"}
            },
            "required"  : ["location"],
        },
        "strict"     : True,
        "type"       : "function",
    }


def load_japanese_cities() -> Optional[List[Dict[str, Any]]]:
    """日本の都市データ読み込み"""
    cities_path = config.get("paths.cities_csv", "data/cities_list.csv")
    try:
        import pandas as pd
        df = pd.read_csv(cities_path)
        jp_cities = df[df["country"] == "Japan"][["name", "lat", "lon"]].to_dict('records')
        return jp_cities
    except Exception as e:
        logger.error(f"都市データ読み込みエラー: {e}")
        return None


# ==================================================
# メインデモクラス
# ==================================================
class ResponsesParseDemo:
    """Responses API包括デモクラス"""

    def __init__(self):
        self.demo_name = "responses_parse_comprehensive"
        self.message_manager = MessageManagerUI(f"messages_{self.demo_name}")
        SessionStateManager.init_session_state()

    def setup_sidebar(self, selected_model: str):
        """サイドバーの設定"""
        st.sidebar.write("情報パネル")

        # 各情報パネルを表示
        InfoPanelManager.show_api_info()
        InfoPanelManager.show_model_capabilities(selected_model)
        InfoPanelManager.show_demo_guide()

        # 設定パネル
        UIHelper.show_settings_panel()

    @error_handler_ui
    @timer_ui
    def demo_parse_basic(self, selected_model: str):
        """01_00: responses.parseの基本"""
        st.subheader("responses.parse 基本デモ")

        st.info("""
        **responses.parse**は構造化された出力を生成する基本機能です。
        テキストから人物情報を抽出してPydanticモデルで受け取ります。
        """)

        # サンプルテキスト
        sample_text = config.get("samples.prompts.event_example",
                                 "私の名前は田中太郎、30歳、東京在住です。友人は鈴木健太、28歳、大阪在住です。")

        user_input, submitted = UIHelper.create_input_form(
            key="parse_basic_form",
            label="人物情報を含むテキストを入力してください",
            submit_label="構造化",
            value=sample_text,
            help="名前、年齢、住所が含まれるテキストを入力"
        )

        # スキーマ表示
        with st.expander("出力スキーマ", expanded=False):
            st.code("""
class UserInfo(BaseModel):
    name: str = Field(..., description="ユーザー名")
    age: int = Field(..., ge=0, le=150, description="年齢") 
    city: str = Field(..., description="居住都市")

class People(BaseModel):
    users: List[UserInfo] = Field(..., description="ユーザー一覧")
    total_count: int = Field(..., description="総人数")
            """, language="python")

        if submitted and user_input:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("メッセージを準備中...")
                progress_bar.progress(25)

                messages = [
                    EasyInputMessageParam(
                        role="developer",
                        content="あなたは情報抽出の専門家です。テキストから人物情報を抽出してください。"
                    ),
                    EasyInputMessageParam(
                        role="user",
                        content=[ResponseInputTextParam(type="input_text", text=user_input)]
                    )
                ]

                status_text.text("構造化データを生成中...")
                progress_bar.progress(70)

                client = OpenAIClient()
                response = client.client.responses.parse(
                    model=selected_model,
                    input=messages,
                    text_format=People
                )

                status_text.text("構造化完了!")
                progress_bar.progress(100)

                # 結果表示
                if hasattr(response, 'output_parsed'):
                    people: People = response.output_parsed

                    st.success(f"🎉 {people.total_count}人の情報を抽出しました!")

                    # 人物情報表示
                    for i, person in enumerate(people.users, 1):
                        with st.container():
                            st.write(f"**Person {i}**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write("名前:", person.name)
                            with col2:
                                st.write("年齢:", f"{person.age}歳")
                            with col3:
                                st.write("居住地:", person.city)
                            st.divider()

                    # JSON出力
                    with st.expander("📊 JSON出力", expanded=False):
                        st.json(people.model_dump())

                        UIHelper.create_download_button(
                            people.model_dump(),
                            "extracted_people.json",
                            "application/json",
                            "JSONダウンロード"
                        )

            except Exception as e:
                st.error(f"構造化エラー: {str(e)}")
                logger.error(f"Parse basic error: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

    @error_handler_ui
    @timer_ui
    def demo_create_basic(self, selected_model: str):
        """01_01: responses.create 基本"""
        st.subheader("responses.create 基本デモ")

        st.info("""
        **responses.create**は最も基本的なAPI呼び出しです。
        自然言語での質問に対してモデルが回答します。
        """)

        user_input, submitted = UIHelper.create_input_form(
            key="create_basic_form",
            label="質問を入力してください",
            submit_label="送信",
            placeholder="例: OpenAIのAPIについて教えて",
            help="何でも気軽に質問してください"
        )

        if submitted and user_input:
            # トークン数チェック
            token_count = TokenManager.count_tokens(user_input, selected_model)
            UIHelper.show_token_info(user_input, selected_model)

            limits = TokenManager.get_model_limits(selected_model)
            if token_count > limits['max_tokens'] * 0.8:
                st.warning(f"入力が長すぎます ({token_count:,} トークン)")
                return

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("メッセージを準備中...")
                progress_bar.progress(20)

                messages = self.message_manager.get_default_messages()
                messages.append(EasyInputMessageParam(role="user", content=user_input))

                status_text.text("AIが回答を生成中...")
                progress_bar.progress(50)

                client = OpenAIClient()
                response = client.create_response(messages, model=selected_model)

                status_text.text("完了!")
                progress_bar.progress(100)

                # 回答表示
                ResponseProcessorUI.display_response(response, show_details=True)

                # メッセージ履歴に追加
                self.message_manager.add_message("user", user_input)
                texts = ResponseProcessorUI.extract_text(response)
                if texts:
                    self.message_manager.add_message("assistant", texts[0])

            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
                logger.error(f"Create basic error: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

    @error_handler_ui
    @timer_ui
    def demo_memory_conversation(self, selected_model: str):
        """01_011: 履歴付き会話"""
        st.subheader("履歴付き会話デモ")

        st.info("""
        **履歴付き会話**では前の会話内容を記憶して連続的な対話が可能です。
        """)

        # 現在の履歴表示
        messages = self.message_manager.get_messages()
        if len(messages) > 3:  # デフォルトメッセージ以外がある場合
            with st.expander("会話履歴", expanded=True):
                UIHelper.display_messages(messages, show_system=False)

        user_input, submitted = UIHelper.create_input_form(
            key="memory_form",
            label="続きの質問を入力してください",
            submit_label="送信",
            help="前の会話を覚えているので、続きの質問ができます"
        )

        if submitted and user_input:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("会話履歴を含めて処理中...")
                progress_bar.progress(30)

                # 履歴にユーザーメッセージを追加
                self.message_manager.add_message("user", user_input)

                status_text.text("AIが回答を生成中...")
                progress_bar.progress(70)

                # 全履歴でAPI呼び出し
                client = OpenAIClient()
                response = client.create_response(
                    self.message_manager.get_messages(),
                    model=selected_model
                )

                status_text.text("完了!")
                progress_bar.progress(100)

                # 回答表示
                ResponseProcessorUI.display_response(response, show_details=True)

                # アシスタント回答を履歴に追加
                texts = ResponseProcessorUI.extract_text(response)
                if texts:
                    self.message_manager.add_message("assistant", texts[0])

                st.rerun()

            except Exception as e:
                st.error(f"会話エラー: {str(e)}")
                logger.error(f"Memory conversation error: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

        # 履歴管理
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 履歴クリア", key="memory_clear"):
                self.message_manager.clear_messages()
                st.rerun()
        with col2:
            message_count = len(messages) - 3  # デフォルトメッセージを除く
            st.metric("会話数", max(0, message_count))

    @error_handler_ui
    @timer_ui
    def demo_image_url(self, selected_model: str):
        """01_02: 画像入力(URL)"""
        st.subheader("画像入力(URL)デモ")

        st.info("""
        **画像URL入力**でWeb上の画像を解析できます。
        画像の内容を理解して質問に回答します。
        """)

        # デフォルト画像URL
        default_url = config.get("samples.images.nature",
                                 "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg")

        col1, col2 = st.columns([3, 1])
        with col1:
            image_url = st.text_input(
                "画像URLを入力してください",
                value=default_url,
                help="解析したい画像のURLを指定"
            )
        with col2:
            question = st.text_input(
                "質問内容",
                value="この画像を説明してください",
                help="画像に関する質問"
            )

        # 画像プレビュー
        if image_url:
            try:
                st.image(image_url, caption="解析対象画像", width=400)
            except Exception:
                st.warning("画像のプレビューに失敗しました")

        if st.button("画像解析実行", key="image_url_analyze"):
            if not image_url:
                st.error("画像URLを入力してください")
                return

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("画像を読み込み中...")
                progress_bar.progress(30)

                messages = [
                    EasyInputMessageParam(
                        role="user",
                        content=[
                            ResponseInputTextParam(type="input_text", text=question),
                            ResponseInputImageParam(type="input_image", image_url=image_url, detail="auto")
                        ]
                    )
                ]

                status_text.text("AIが画像を解析中...")
                progress_bar.progress(70)

                client = OpenAIClient()
                response = client.create_response(messages, model=selected_model)

                status_text.text("解析完了!")
                progress_bar.progress(100)

                # 結果表示
                ResponseProcessorUI.display_response(response, show_details=True)

            except Exception as e:
                st.error(f"画像解析エラー: {str(e)}")
                logger.error(f"Image URL analysis error: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

    @error_handler_ui
    @timer_ui
    def demo_image_base64(self, selected_model: str):
        """01_021: 画像入力(Base64)"""
        st.subheader("画像入力(Base64)デモ")

        st.info("""
        **ローカル画像ファイル**をアップロードして解析できます。
        Base64エンコードして送信します。
        """)

        # ファイルアップローダー
        uploaded_file = st.file_uploader(
            "画像ファイルを選択してください",
            type=['png', 'jpg', 'jpeg', 'webp', 'gif'],
            help="対応形式: PNG, JPG, JPEG, WebP, GIF"
        )

        question = st.text_input(
            "質問内容",
            value="この画像について詳しく説明してください",
            key="base64_question"
        )

        if uploaded_file and question:
            # 画像プレビュー
            st.image(uploaded_file, caption="アップロード画像", width=400)

            if st.button("画像解析実行", key="image_base64_analyze"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    status_text.text("ファイルを読み込み中...")
                    progress_bar.progress(20)

                    # Base64エンコード
                    image_bytes = uploaded_file.read()
                    b64_string = base64.b64encode(image_bytes).decode()

                    status_text.text("Base64エンコード完了...")
                    progress_bar.progress(40)

                    # データURLを作成
                    mime_type = f"image/{uploaded_file.type.split('/')[-1]}"
                    data_url = f"data:{mime_type};base64,{b64_string}"

                    messages = [
                        EasyInputMessageParam(
                            role="user",
                            content=[
                                ResponseInputTextParam(type="input_text", text=question),
                                ResponseInputImageParam(type="input_image", image_url=data_url, detail="auto")
                            ]
                        )
                    ]

                    status_text.text("🤖 AI が画像を解析中...")
                    progress_bar.progress(70)

                    client = OpenAIClient()
                    response = client.create_response(messages, model=selected_model)

                    status_text.text("解析完了!")
                    progress_bar.progress(100)

                    # 結果表示
                    ResponseProcessorUI.display_response(response, show_details=True)

                except Exception as e:
                    st.error(f"画像解析エラー: {str(e)}")
                    logger.error(f"Image base64 analysis error: {e}")
                finally:
                    progress_bar.empty()
                    status_text.empty()

    @error_handler_ui
    @timer_ui
    def demo_structured_output(self, selected_model: str):
        """01_03: 構造化出力"""
        st.subheader("構造化出力デモ")

        st.info("""
        **構造化出力**では事前に定義したJSONスキーマに従って
        モデルの出力を構造化できます。
        """)

        # サンプルテキスト
        sample_text = "台湾フェス2025 ～あつまれ！究極の台湾グルメ～ in Kawasaki Spark（5/3・5/4開催）参加者：王さん、林さん、佐藤さん"

        user_input, submitted = UIHelper.create_input_form(
            key="structured_form",
            label="イベント詳細を入力してください",
            submit_label="構造化",
            value=sample_text,
            help="イベント名、日付、参加者が含まれるテキスト"
        )

        # スキーマ表示
        with st.expander("JSONスキーマ", expanded=False):
            schema = {
                "type"                : "object",
                "properties"          : {
                    "name"        : {"type": "string"},
                    "date"        : {"type": "string"},
                    "participants": {"type": "array", "items": {"type": "string"}}
                },
                "required"            : ["name", "date", "participants"],
                "additionalProperties": False
            }
            st.json(schema)

        if submitted and user_input:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("構造化プロンプトを準備中...")
                progress_bar.progress(25)

                messages = [
                    EasyInputMessageParam(role="developer", content="Extract event details from the text."),
                    EasyInputMessageParam(role="user",
                                          content=[ResponseInputTextParam(type="input_text", text=user_input)])
                ]

                # 構造化設定
                schema = {
                    "type"                : "object",
                    "properties"          : {
                        "name"        : {"type": "string"},
                        "date"        : {"type": "string"},
                        "participants": {"type": "array", "items": {"type": "string"}}
                    },
                    "required"            : ["name", "date", "participants"],
                    "additionalProperties": False
                }

                text_cfg = ResponseTextConfigParam(
                    format=ResponseFormatTextJSONSchemaConfigParam(
                        name="event_extraction",
                        type="json_schema",
                        schema=schema,
                        strict=True,
                    )
                )

                status_text.text("構造化データを生成中...")
                progress_bar.progress(70)

                client = OpenAIClient()
                response = client.client.responses.create(
                    model=selected_model,
                    input=messages,
                    text=text_cfg
                )

                status_text.text("構造化完了!")
                progress_bar.progress(100)

                # 結果処理
                try:
                    event_data = json.loads(response.output_text)
                    event = Event.model_validate(event_data)

                    st.success("イベント情報を構造化しました!")

                    # 構造化データ表示
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.write("**イベント情報**")
                        st.write(f"**名称**: {event.name}")
                        st.write(f"**日程**: {event.date}")
                        st.write("**参加者**:")
                        for participant in event.participants:
                            st.write(f"  - {participant}")

                    with col2:
                        st.write("**📊 JSON出力**")
                        st.json(event.model_dump())

                    # ダウンロード
                    UIHelper.create_download_button(
                        event.model_dump(),
                        "event_data.json",
                        "application/json",
                        "📥 JSONダウンロード"
                    )

                except Exception as parse_error:
                    st.error(f"構造化データの解析に失敗: {parse_error}")
                    st.write("Raw response:")
                    st.code(response.output_text)

            except Exception as e:
                st.error(f"構造化エラー: {str(e)}")
                logger.error(f"Structured output error: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

    @error_handler_ui
    @timer_ui
    def demo_function_calling(self, selected_model: str):
        """01_04: 関数呼び出し"""
        st.subheader("関数呼び出しデモ")

        st.info("""
        **Function Calling**では外部の関数をモデルが自動的に呼び出せます。
        ここでは天気情報取得の例を示します。
        """)

        # 都市選択
        cities_data = load_japanese_cities()
        if not cities_data:
            st.error("都市データの読み込みに失敗しました")
            return

        city_names = [city["name"] for city in cities_data]
        selected_city = st.selectbox(
            "都市を選択してください",
            city_names,
            help="天気を取得したい都市を選択"
        )

        # 関数ツール定義表示
        with st.expander("🔧 関数定義", expanded=False):
            function_tool = get_weather_function_tool()
            st.json(function_tool)

        user_query = st.text_input(
            "天気に関する質問",
            value=f"{selected_city}の天気はどうですか？",
            help="天気について自然言語で質問"
        )

        if st.button("🌤️ 天気取得実行", key="function_call"):
            if not user_query:
                st.error("質問を入力してください")
                return

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("関数ツールを準備中...")
                progress_bar.progress(20)

                # 関数ツール定義
                function_tool = get_weather_function_tool()

                messages = [
                    EasyInputMessageParam(
                        role="developer",
                        content="あなたは天気情報を提供するアシスタントです。必要に応じて関数を呼び出してください。"
                    ),
                    EasyInputMessageParam(role="user", content=user_query)
                ]

                status_text.text("AI が関数呼び出しを判断中...")
                progress_bar.progress(60)

                client = OpenAIClient()
                response = client.client.responses.create(
                    model=selected_model,
                    input=messages,
                    tools=[function_tool]
                )

                status_text.text("関数呼び出し完了!")
                progress_bar.progress(100)

                # 結果表示
                ResponseProcessorUI.display_response(response, show_details=True)

                # 関数呼び出し詳細
                if hasattr(response, 'output'):
                    for output in response.output:
                        if hasattr(output, 'type') and output.type == 'function_call':
                            st.write("**🔧 関数呼び出し詳細**")
                            st.write(f"- 関数名: `{output.name}`")
                            if hasattr(output, 'arguments'):
                                st.write(f"- 引数: `{output.arguments}`")

            except Exception as e:
                st.error(f"関数呼び出しエラー: {str(e)}")
                logger.error(f"Function calling error: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

    @error_handler_ui
    @timer_ui
    def demo_web_search(self, selected_model: str):
        """01_062: Web検索"""
        st.subheader("Web検索デモ")

        st.info("""
        **Web Search**では最新のWeb情報を検索して回答に反映できます。
        リアルタイムの情報取得が可能です。
        """)

        # 検索設定
        col1, col2 = st.columns([2, 1])
        with col1:
            search_query = st.text_input(
                "検索クエリを入力してください",
                value=config.get("samples.prompts.weather_query", "今日の東京の天気"),
                help="検索したい内容を自然言語で入力"
            )
        with col2:
            search_size = st.selectbox(
                "検索サイズ",
                ["low", "medium", "high"],
                index=1,
                help="検索結果の詳細度"
            )

        if st.button("Web検索実行", key="web_search"):
            if not search_query:
                st.error("検索クエリを入力してください")
                return

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("Web検索ツールを準備中...")
                progress_bar.progress(20)

                # Web検索ツール設定
                user_location = UserLocation(
                    type="approximate",
                    country="JP",
                    city="Tokyo",
                    region="Tokyo"
                )

                ws_tool = WebSearchToolParam(
                    type="web_search_preview",
                    user_location=user_location,
                    search_context_size=search_size
                )

                status_text.text("Web検索を実行中...")
                progress_bar.progress(60)

                client = OpenAIClient()
                response = client.client.responses.create(
                    model=selected_model,
                    tools=[ws_tool],
                    input=search_query
                )

                status_text.text("検索完了!")
                progress_bar.progress(100)

                # 結果表示
                ResponseProcessorUI.display_response(response, show_details=True)

            except Exception as e:
                st.error(f"Web検索エラー: {str(e)}")
                logger.error(f"Web search error: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

    def show_message_history(self):
        """メッセージ履歴表示"""
        st.subheader("会話履歴")

        messages = self.message_manager.get_messages()
        if messages:
            # 履歴表示
            UIHelper.display_messages(messages, show_system=True)

            # 履歴操作
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("履歴クリア"):
                    self.message_manager.clear_messages()
                    st.rerun()
            with col2:
                if st.button("履歴エクスポート"):
                    export_data = self.message_manager.export_messages_ui()
                    UIHelper.create_download_button(
                        export_data,
                        "chat_history.json",
                        "application/json",
                        "保存"
                    )
            with col3:
                message_count = len(messages) - 3  # デフォルトメッセージを除く
                st.write("会話数", max(0, message_count))
            with col4:
                total_tokens = sum(TokenManager.count_tokens(str(msg.get("content", ""))) for msg in messages)
                st.write("総トークン", f"{total_tokens:,}")
        else:
            st.info("会話履歴がありません")

    def run(self):
        """メインデモ実行"""
        # ページ初期化
        init_page("Responses API デモ", sidebar_title="📋 情報パネル")

        # モデル選択
        selected_model = select_model(self.demo_name)

        # サイドバー設定
        self.setup_sidebar(selected_model)

        # メイン画面
        st.markdown("""
        #### 概要
        OpenAI Responses APIの包括的なデモアプリケーションです。
        基本機能から高度な機能まで様々なユースケースを体験できます。
        """)

        # タブでデモを分離
        tabs = st.tabs([
            "Parse Basic",
            "Create",
            "Memory",
            "Image URL",
            "Image File",
            "Structured",
            "Function",
            "Web Search",
            "履歴"
        ])

        with tabs[0]:
            self.demo_parse_basic(selected_model)

        with tabs[1]:
            self.demo_create_basic(selected_model)

        with tabs[2]:
            self.demo_memory_conversation(selected_model)

        with tabs[3]:
            self.demo_image_url(selected_model)

        with tabs[4]:
            self.demo_image_base64(selected_model)

        with tabs[5]:
            self.demo_structured_output(selected_model)

        with tabs[6]:
            self.demo_function_calling(selected_model)

        with tabs[7]:
            self.demo_web_search(selected_model)

        with tabs[8]:
            self.show_message_history()

        # フッター
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray;'>
        新しいヘルパーモジュールを使用 | 
        左ペインで詳細情報を確認できます |
        OpenAI Responses APIデモ
        </div>
        """, unsafe_allow_html=True)


# ==================================================
# メイン実行部
# ==================================================
def main():
    """メイン関数"""
    try:
        demo = ResponsesParseDemo()
        demo.run()
    except Exception as e:
        st.error(f"アプリケーションエラー: {str(e)}")
        logger.error(f"Application error: {e}")

        if config.get("experimental.debug_mode", False):
            st.exception(e)


if __name__ == "__main__":
    main()

# streamlit run a20_01_responses_parse.py --server.port=8501
