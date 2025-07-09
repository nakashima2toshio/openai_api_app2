# streamlit run a20_02_responses_tools_pydantic_parse.py --server.port=8503
# pip install --upgrade openai
# ---------------------------------------------------- 情報：
# https://cookbook.openai.com/examples/structured_outputs_intro
# 基本的には、Responses.parseを利用するのがおすすめ
# ----------------------------------------------------
# [Cookbook ] https://cookbook.openai.com/
# [API      ]  https://github.com/openai/openai-python
# [Agent SDK] https://github.com/openai/openai-agents-python
# --- --------------
# [Model] https://platform.openai.com/docs/pricing
# ----------------------------------------------------
#
### （2）tools param 概要・一覧
# | 関数                              | 目的・概要                                          |
# | ------------------------------- | ---------------------------------------------- |
# | `web_search_tool_param`         | インターネット検索を実行し、取得記事をモデルに渡して最新情報を回答に反映させる。       |
# | `function_tool_param_by_schema` | モデルが外部API（為替レート取得）を安全に自動呼び出しし、結果を回答へ組み込む。      |
# | `file_search_tool_param`        | 自前ベクトルストアを意味検索し、関連文書を引用して回答する（RAG機能）。          |
# | `computer_use_tool_param`       | 仮想PC/ブラウザ環境をAIが操作するRPA機能。操作結果やスクリーンショットを取得できる。 |
# | `structured_output_by_schema`   | モデル出力をユーザ定義JSONスキーマへ厳密整形し、機械可読な構造化データとして取得。    |
# | `image_param`                   | Vision機能。画像＋質問を送り、画像内容を理解・回答させるサンプル。           |

# a10_01_responses_tools_pydantic_parse.py
# ----------------------------------------------------
# [サンプル01] toolsの使い方
# (01_01) 基本的なfunction_callのstructured output
# (01_02) 複数ツールの登録・複数関数呼び出し
# (01_21) 複数ツールの登録・複数関数呼び出し
# (01_03) ユーザー独自の複雑な構造体（入れ子あり）
# (01_04) Enum型や型安全なオプションパラメータ付き
# (01_05) text_format引数で自然文のstructured outputを生成
# ----------------------------------------------------
# [サンプル02] 構造化データ抽出
# (02_01) 基本パターン （シンプルな構造化データ抽出）
# (02_011) 基本パターン（複数の構造化データ抽出）
# (02_02) 複雑なクエリパターン（条件・ソートなど）
# (02_03) 列挙型・動的な値の利用パターン
# (02_04) 階層化された出力構造（Nested Structure）
# (02_05) 会話履歴を持った連続した構造化出力の処理
# -----------------------------------------------
# streamlit run a20_02_responses_tools_pydantic_parse.py --server.port=8503
# 改修版: 新しいヘルパーモジュールを使用したTools & Pydantic Parse デモ
# ==================================================
# OpenAI Responses APIのtools paramとPydantic構造化出力の包括デモ
# - 基本的なfunction_callのstructured output
# - 複数ツールの登録・複数関数呼び出し
# - 複雑な構造体（入れ子あり）
# - Enum型や型安全なオプションパラメータ
# - text_format引数による自然文のstructured output
# - 構造化データ抽出（基本〜複雑）
# - 会話履歴を持った連続構造化出力
# ==================================================

import os
import sys
import json
import time
import requests
import pprint
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pathlib import Path

import streamlit as st
from pydantic import BaseModel, Field
from openai import OpenAI
from openai import pydantic_function_tool
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputTextParam,
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
        safe_json_dumps
    )
except ImportError as e:
    st.error(f"ヘルパーモジュールのインポートに失敗しました: {e}")
    st.stop()


# ==================================================
# Pydantic モデル定義
# ==================================================

# サンプル01用モデル
class WeatherRequest(BaseModel):
    city: str = Field(..., description="都市名")
    date: str = Field(..., description="日付")


class NewsRequest(BaseModel):
    topic: str = Field(..., description="ニューストピック")
    date: str = Field(..., description="日付")


class CalculatorRequest(BaseModel):
    exp: str = Field(..., description="計算式（例: 2+2）")


class FAQSearchRequest(BaseModel):
    query: str = Field(..., description="FAQ検索クエリ")


class Task(BaseModel):
    name: str = Field(..., description="タスク名")
    deadline: str = Field(..., description="期限")


class ProjectRequest(BaseModel):
    project_name: str = Field(..., description="プロジェクト名")
    tasks: List[Task] = Field(..., description="タスクリスト")


class Unit(str, Enum):
    celsius = "celsius"
    fahrenheit = "fahrenheit"


class WeatherRequestWithUnit(BaseModel):
    city: str = Field(..., description="都市名")
    date: str = Field(..., description="日付")
    unit: Unit = Field(..., description="温度単位")


# サンプル02用モデル
class PersonInfo(BaseModel):
    name: str = Field(..., description="名前")
    age: int = Field(..., ge=0, le=150, description="年齢")


class BookInfo(BaseModel):
    title: str = Field(..., description="書籍タイトル")
    author: str = Field(..., description="著者")
    year: int = Field(..., description="出版年")


class ExtractedData(BaseModel):
    persons: List[PersonInfo] = Field(default_factory=list, description="人物リスト")
    books: List[BookInfo] = Field(default_factory=list, description="書籍リスト")


class Operator(str, Enum):
    eq = "="
    ne = "!="
    gt = ">"
    lt = "<"


class Condition(BaseModel):
    column: str = Field(..., description="カラム名")
    operator: Operator = Field(..., description="演算子")
    value: Union[str, int] = Field(..., description="値")


class Query(BaseModel):
    table: str = Field(..., description="テーブル名")
    conditions: List[Condition] = Field(..., description="条件リスト")
    sort_by: str = Field(..., description="ソートキー")
    ascending: bool = Field(..., description="昇順フラグ")


class Priority(str, Enum):
    high = "高"
    medium = "中"
    low = "低"


class TaskWithPriority(BaseModel):
    description: str = Field(..., description="タスク説明")
    priority: Priority = Field(..., description="優先度")


class Step(BaseModel):
    explanation: str = Field(..., description="ステップの説明")
    output: str = Field(..., description="ステップの出力")


class MathSolution(BaseModel):
    steps: List[Step] = Field(..., description="解決ステップ")
    answer: str = Field(..., description="最終回答")


class QAResponse(BaseModel):
    question: str = Field(..., description="質問")
    answer: str = Field(..., description="回答")


# ==================================================
# 情報パネル管理
# ==================================================
class ToolsPanelManager:
    """左ペインの情報パネル管理"""

    @staticmethod
    def show_tools_overview():
        """Tools概要パネル"""
        with st.sidebar.expander("🔧 Tools 概要", expanded=True):
            st.write("**利用可能なツール**")
            tools_info = [
                "🔍 **function_tool_param** - 外部関数呼び出し",
                "📊 **structured_output** - 構造化データ生成",
                "🔗 **pydantic_function_tool** - 型安全な関数定義",
                "💬 **text_format** - 自然言語構造化",
                "🔄 **parse API** - 直接構造化出力"
            ]
            for tool in tools_info:
                st.write(tool)

    @staticmethod
    def show_pydantic_benefits():
        """Pydantic利点説明"""
        with st.sidebar.expander("✨ Pydantic の利点", expanded=False):
            st.write("**型安全性**")
            st.write("- 自動バリデーション")
            st.write("- エラー検出")
            st.write("- IDE補完")

            st.write("**保守性**")
            st.write("- 明確なスキーマ定義")
            st.write("- 自動ドキュメント生成")
            st.write("- 再利用可能")

            st.write("**開発効率**")
            st.write("- 手動JSON Schema不要")
            st.write("- 型チェック")
            st.write("- 変換処理自動化")

    @staticmethod
    def show_demo_structure():
        """デモ構成説明"""
        with st.sidebar.expander("📚 デモ構成", expanded=False):
            st.write("**サンプル01 - Tools使用方法**")
            st.write("- 基本的なfunction call")
            st.write("- 複数ツール呼び出し")
            st.write("- 複雑な構造体")
            st.write("- Enum型対応")
            st.write("- 自然言語構造化")

            st.write("**サンプル02 - データ抽出**")
            st.write("- シンプル抽出")
            st.write("- 複数エンティティ")
            st.write("- 複雑クエリ")
            st.write("- 列挙型利用")
            st.write("- 階層構造")
            st.write("- 会話履歴管理")

    @staticmethod
    def show_performance_tips():
        """パフォーマンスヒント"""
        with st.sidebar.expander("⚡ パフォーマンスヒント", expanded=False):
            st.write("**モデル選択**")
            st.write("- 構造化: gpt-4o, gpt-4.1推奨")
            st.write("- 速度重視: gpt-4o-mini")
            st.write("- 推論: o1, o3シリーズ")

            st.write("**プロンプト最適化**")
            st.write("- 明確な指示")
            st.write("- 例示の活用")
            st.write("- スキーマの簡潔性")


# ==================================================
# ユーティリティ関数
# ==================================================
def get_default_messages_with_developer(content: str = None) -> List[EasyInputMessageParam]:
    """開発者メッセージ付きデフォルトメッセージ"""
    messages = []

    if content:
        messages.append(EasyInputMessageParam(
            role="developer",
            content=content
        ))

    messages.extend([
        EasyInputMessageParam(
            role="user",
            content="APIの使用方法を教えてください"
        ),
        EasyInputMessageParam(
            role="assistant",
            content="OpenAI APIを使用したツール呼び出しと構造化出力についてご説明します。"
        )
    ])

    return messages


def safe_api_call(func, *args, **kwargs):
    """安全なAPI呼び出し"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"API call failed: {e}")
        raise


# ==================================================
# メインデモクラス
# ==================================================
class ToolsPydanticDemo:
    """Tools & Pydantic Parse 包括デモクラス"""

    def __init__(self):
        self.demo_name = "tools_pydantic_comprehensive"
        self.message_manager = MessageManagerUI(f"messages_{self.demo_name}")
        self.client = OpenAI()
        SessionStateManager.init_session_state()

    def setup_sidebar(self, selected_model: str):
        """サイドバーの設定"""
        st.sidebar.write("📋 情報パネル")

        # 各情報パネルを表示
        ToolsPanelManager.show_tools_overview()
        ToolsPanelManager.show_pydantic_benefits()
        ToolsPanelManager.show_demo_structure()
        ToolsPanelManager.show_performance_tips()

        # 設定パネル
        UIHelper.show_settings_panel()

    @error_handler_ui
    @timer_ui
    def sample_01_01_basic_function_call(self, selected_model: str):
        """01_01: 基本的なfunction_callのstructured output"""
        st.subheader("🔧 基本的なFunction Call")

        st.info("""
        **基本的なFunction Call**では複数のツールを登録して、
        AIが適切なツールを選択・実行します。
        """)

        # 入力フォーム
        user_input, submitted = UIHelper.create_input_form(
            key="basic_function_form",
            label="要求を入力してください",
            submit_label="🚀 実行",
            value="東京と大阪の明日の天気と、AIの最新ニュースを教えて",
            help="天気やニュースに関する要求を入力"
        )

        # ツール定義表示
        with st.expander("🔧 登録ツール", expanded=False):
            st.write("**WeatherRequest**: 天気情報取得")
            st.code("city: str, date: str", language="python")
            st.write("**NewsRequest**: ニュース情報取得")
            st.code("topic: str, date: str", language="python")

        if submitted and user_input:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("🔧 ツールを準備中...")
                progress_bar.progress(25)

                messages = get_default_messages_with_developer(
                    "あなたは天気とニュースの情報を提供するアシスタントです。適切なツールを使用してください。"
                )
                messages.append(EasyInputMessageParam(
                    role="user",
                    content=[ResponseInputTextParam(type="input_text", text=user_input)]
                ))

                status_text.text("🤖 AIがツールを選択中...")
                progress_bar.progress(60)

                response = self.client.responses.parse(
                    model=selected_model,
                    input=messages,
                    tools=[
                        pydantic_function_tool(WeatherRequest),
                        pydantic_function_tool(NewsRequest)
                    ]
                )

                status_text.text("✅ 実行完了!")
                progress_bar.progress(100)

                # 結果表示
                st.subheader("🎯 実行結果")

                if hasattr(response, 'output') and response.output:
                    for i, function_call in enumerate(response.output, 1):
                        if hasattr(function_call, 'name') and hasattr(function_call, 'parsed_arguments'):
                            st.write(f"**ツール {i}: {function_call.name}**")

                            # 引数表示
                            args = function_call.parsed_arguments
                            if hasattr(args, 'model_dump'):
                                args_dict = args.model_dump()
                            else:
                                args_dict = args.__dict__ if hasattr(args, '__dict__') else str(args)

                            st.json(args_dict)

                            # 模擬実行結果
                            if function_call.name == "WeatherRequest":
                                st.success(f"📊 天気データ取得: {args_dict.get('city', 'N/A')}")
                            elif function_call.name == "NewsRequest":
                                st.success(f"📰 ニュースデータ取得: {args_dict.get('topic', 'N/A')}")

                            st.divider()
                else:
                    st.warning("ツール呼び出しが検出されませんでした")

                # 詳細情報
                with st.expander("📊 詳細情報", expanded=False):
                    ResponseProcessorUI.display_response(response, show_details=True)

            except Exception as e:
                st.error(f"Function call エラー: {str(e)}")
                logger.error(f"Basic function call error: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

    @error_handler_ui
    @timer_ui
    def sample_01_021_multiple_tools(self, selected_model: str):
        """01_021: 複数ツールの登録・複数関数呼び出し"""
        st.subheader("🔄 複数ツール呼び出し")

        st.info("""
        **複数ツール**を同時に登録して、AIが必要に応じて
        複数のツールを組み合わせて使用します。
        """)

        user_input, submitted = UIHelper.create_input_form(
            key="multiple_tools_form",
            label="複合要求を入力してください",
            submit_label="🔄 実行",
            value="2+2はいくつですか？またはFAQから確認してください。",
            help="計算とFAQ検索の複合要求"
        )

        # ツール説明
        with st.expander("🔧 利用可能ツール", expanded=False):
            tools_desc = {
                "CalculatorRequest": "数式計算（exp: str）",
                "FAQSearchRequest" : "FAQ検索（query: str）"
            }
            for tool, desc in tools_desc.items():
                st.write(f"**{tool}**: {desc}")

        if submitted and user_input:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("🔧 複数ツールを準備中...")
                progress_bar.progress(20)

                messages = get_default_messages_with_developer(
                    "あなたは計算とFAQ検索ができるアシスタントです。"
                )
                messages.append(EasyInputMessageParam(
                    role="user",
                    content=[ResponseInputTextParam(type="input_text", text=user_input)]
                ))

                status_text.text("🤖 AIがツールを分析中...")
                progress_bar.progress(60)

                response = self.client.responses.parse(
                    model=selected_model,
                    input=messages,
                    tools=[
                        pydantic_function_tool(CalculatorRequest, name="calculator"),
                        pydantic_function_tool(FAQSearchRequest, name="faq_search"),
                    ],
                )

                status_text.text("✅ 実行完了!")
                progress_bar.progress(100)

                # 結果処理と表示
                st.subheader("🎯 ツール実行結果")

                if hasattr(response, 'output') and response.output:
                    for i, function_call in enumerate(response.output, 1):
                        if hasattr(function_call, 'name') and hasattr(function_call, 'parsed_arguments'):
                            st.write(f"**実行 {i}: {function_call.name}**")

                            args = function_call.parsed_arguments
                            args_dict = args.model_dump() if hasattr(args, 'model_dump') else str(args)

                            # ツール別の模擬実行
                            if function_call.name == "calculator":
                                exp = args_dict.get('exp', '')
                                try:
                                    result = str(eval(exp)) if exp else "計算式が不明"
                                    st.success(f"🧮 計算結果: {exp} = {result}")
                                except:
                                    st.error(f"❌ 計算エラー: {exp}")

                            elif function_call.name == "faq_search":
                                query = args_dict.get('query', '')
                                st.success(f"❓ FAQ検索: 「{query}」の回答を検索中...")
                                st.info("模擬回答: こちらがFAQの検索結果です。")

                            # 引数詳細
                            with st.expander(f"引数詳細 - {function_call.name}", expanded=False):
                                st.json(args_dict)

                            st.divider()

            except Exception as e:
                st.error(f"Multiple tools エラー: {str(e)}")
                logger.error(f"Multiple tools error: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

    @error_handler_ui
    @timer_ui
    def sample_01_03_complex_structure(self, selected_model: str):
        """01_03: ユーザー独自の複雑な構造体（入れ子あり）"""
        st.subheader("🏗️ 複雑な構造体")

        st.info("""
        **入れ子構造**を持つ複雑なPydanticモデルを使用して、
        階層的なデータ構造を扱います。
        """)

        user_input, submitted = UIHelper.create_input_form(
            key="complex_structure_form",
            label="プロジェクト情報を入力してください",
            submit_label="🏗️ 構造化",
            value="プロジェクト『AI開発』には「設計（明日まで）」「実装（来週まで）」というタスクがある",
            help="プロジェクト名とタスク情報を含む文章"
        )

        # 構造定義表示
        with st.expander("🏗️ データ構造", expanded=False):
            st.code("""
class Task(BaseModel):
    name: str = Field(..., description="タスク名")
    deadline: str = Field(..., description="期限")

class ProjectRequest(BaseModel):
    project_name: str = Field(..., description="プロジェクト名")
    tasks: List[Task] = Field(..., description="タスクリスト")
            """, language="python")

        if submitted and user_input:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("🏗️ 複雑構造を準備中...")
                progress_bar.progress(30)

                messages = get_default_messages_with_developer(
                    "あなたはプロジェクト管理の専門家です。テキストからプロジェクト情報を抽出してください。"
                )
                messages.append(EasyInputMessageParam(
                    role="user",
                    content=[ResponseInputTextParam(type="input_text", text=user_input)]
                ))

                status_text.text("🤖 構造化分析中...")
                progress_bar.progress(70)

                response = self.client.responses.parse(
                    model=selected_model,
                    input=messages,
                    tools=[pydantic_function_tool(ProjectRequest)]
                )

                status_text.text("✅ 構造化完了!")
                progress_bar.progress(100)

                # 結果表示
                if hasattr(response, 'output') and response.output:
                    function_call = response.output[0]
                    if hasattr(function_call, 'parsed_arguments'):
                        project: ProjectRequest = function_call.parsed_arguments

                        st.success(f"🎉 プロジェクト「{project.project_name}」を構造化しました!")

                        # プロジェクト詳細表示
                        st.subheader("📋 プロジェクト詳細")
                        st.write(f"**プロジェクト名**: {project.project_name}")
                        st.write(f"**タスク数**: {len(project.tasks)}")

                        # タスク一覧
                        st.subheader("📝 タスク一覧")
                        for i, task in enumerate(project.tasks, 1):
                            with st.container():
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.write(f"**{i}. {task.name}**")
                                with col2:
                                    st.write(f"⏰ {task.deadline}")
                                st.divider()

                        # JSON出力
                        with st.expander("📊 JSON出力", expanded=False):
                            st.json(project.model_dump())

            except Exception as e:
                st.error(f"Complex structure エラー: {str(e)}")
                logger.error(f"Complex structure error: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

    @error_handler_ui
    @timer_ui
    def sample_01_04_enum_types(self, selected_model: str):
        """01_04: Enum型や型安全なオプションパラメータ付き"""
        st.subheader("🎯 Enum型対応")

        st.info("""
        **Enum型**を使用して型安全なオプション選択を実現します。
        事前定義された値のみが許可されます。
        """)

        user_input, submitted = UIHelper.create_input_form(
            key="enum_types_form",
            label="天気要求を入力してください",
            submit_label="🌡️ 実行",
            value="ニューヨークの明日の天気を華氏で教えて",
            help="都市名と温度単位を含む要求"
        )

        # Enum定義表示
        with st.expander("🎯 Enum定義", expanded=False):
            st.code("""
class Unit(str, Enum):
    celsius = "celsius"
    fahrenheit = "fahrenheit"

class WeatherRequestWithUnit(BaseModel):
    city: str = Field(..., description="都市名")
    date: str = Field(..., description="日付")
    unit: Unit = Field(..., description="温度単位")
            """, language="python")

        if submitted and user_input:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("🎯 Enum型ツールを準備中...")
                progress_bar.progress(30)

                messages = get_default_messages_with_developer(
                    "あなたは天気情報を提供するアシスタントです。温度単位に注意してください。"
                )
                messages.append(EasyInputMessageParam(
                    role="user",
                    content=[ResponseInputTextParam(type="input_text", text=user_input)]
                ))

                status_text.text("🤖 Enum値を解析中...")
                progress_bar.progress(70)

                response = self.client.responses.parse(
                    model=selected_model,
                    input=messages,
                    tools=[pydantic_function_tool(WeatherRequestWithUnit)]
                )

                status_text.text("✅ Enum処理完了!")
                progress_bar.progress(100)

                # 結果表示
                if hasattr(response, 'output') and response.output:
                    function_call = response.output[0]
                    if hasattr(function_call, 'parsed_arguments'):
                        weather_req: WeatherRequestWithUnit = function_call.parsed_arguments

                        st.success("🎉 Enum型で安全に解析されました!")

                        # 解析結果表示
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🏙️ 都市", weather_req.city)
                        with col2:
                            st.metric("📅 日付", weather_req.date)
                        with col3:
                            unit_display = "摂氏 (°C)" if weather_req.unit == Unit.celsius else "華氏 (°F)"
                            st.metric("🌡️ 単位", unit_display)

                        # Enum値の確認
                        st.write("**🎯 Enum値検証**")
                        st.write(f"- 入力値: `{weather_req.unit.value}`")
                        st.write(f"- Enum型: `{type(weather_req.unit).__name__}`")
                        st.write(f"- 有効値: `{[u.value for u in Unit]}`")

                        # JSON出力
                        with st.expander("📊 構造化データ", expanded=False):
                            st.json(weather_req.model_dump())

            except Exception as e:
                st.error(f"Enum types エラー: {str(e)}")
                logger.error(f"Enum types error: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

    @error_handler_ui
    @timer_ui
    def sample_01_05_text_format(self, selected_model: str):
        """01_05: text_format引数で自然文のstructured outputを生成"""
        st.subheader("📝 自然言語構造化")

        st.info("""
        **text_format**引数を使用して、自然言語の回答を
        構造化されたフォーマットで取得します。
        """)

        user_input, submitted = UIHelper.create_input_form(
            key="text_format_form",
            label="数学問題を入力してください",
            submit_label="🧮 解析",
            value="8x + 31 = 2 を解いてください。途中計算も教えて",
            help="段階的解法が必要な数学問題"
        )

        # 出力フォーマット表示
        with st.expander("📝 出力フォーマット", expanded=False):
            st.code("""
class Step(BaseModel):
    explanation: str = Field(..., description="ステップの説明")
    output: str = Field(..., description="ステップの出力")

class MathSolution(BaseModel):
    steps: List[Step] = Field(..., description="解決ステップ")
    answer: str = Field(..., description="最終回答")
            """, language="python")

        if submitted and user_input:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("📝 自然言語構造化を準備中...")
                progress_bar.progress(25)

                messages = get_default_messages_with_developer(
                    "あなたは数学の家庭教師です。段階的に問題を解いてください。"
                )
                messages.append(EasyInputMessageParam(
                    role="user",
                    content=[ResponseInputTextParam(type="input_text", text=user_input)]
                ))

                status_text.text("🧮 数学的思考を構造化中...")
                progress_bar.progress(70)

                response = self.client.responses.parse(
                    model=selected_model,
                    input=messages,
                    text_format=MathSolution,
                )

                status_text.text("✅ 構造化完了!")
                progress_bar.progress(100)

                # 結果表示
                if hasattr(response, 'output') and response.output:
                    for output in response.output:
                        if hasattr(output, 'type') and output.type == "message":
                            for item in output.content:
                                if hasattr(item, 'type') and item.type == "output_text" and hasattr(item, 'parsed'):
                                    solution: MathSolution = item.parsed

                                    st.success("🎉 数学問題を構造化して解決しました!")

                                    # 解法ステップ表示
                                    st.subheader("📚 解法ステップ")
                                    for i, step in enumerate(solution.steps, 1):
                                        with st.container():
                                            st.write(f"**ステップ {i}**")
                                            st.write(f"**説明**: {step.explanation}")
                                            st.write(f"**結果**: {step.output}")
                                            st.divider()

                                    # 最終回答
                                    st.subheader("🎯 最終回答")
                                    st.success(f"**答え**: {solution.answer}")

                                    # JSON出力
                                    with st.expander("📊 構造化データ", expanded=False):
                                        st.json(solution.model_dump())

            except Exception as e:
                st.error(f"Text format エラー: {str(e)}")
                logger.error(f"Text format error: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

    @error_handler_ui
    @timer_ui
    def sample_02_01_simple_extraction(self, selected_model: str):
        """02_01: 基本パターン（シンプルな構造化データ抽出）"""
        st.subheader("📊 シンプルデータ抽出")

        st.info("""
        **シンプルなデータ抽出**では基本的な情報を
        構造化されたフォーマットで取得します。
        """)

        user_input, submitted = UIHelper.create_input_form(
            key="simple_extraction_form",
            label="人物情報を入力してください",
            submit_label="👤 抽出",
            value="彼女の名前は中島美咲で年齢は27歳です。",
            help="名前と年齢を含む文章"
        )

        # データモデル表示
        with st.expander("👤 データモデル", expanded=False):
            st.code("""
class PersonInfo(BaseModel):
    name: str = Field(..., description="名前")
    age: int = Field(..., description="年齢")
            """, language="python")

        if submitted and user_input:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("📊 データ抽出を準備中...")
                progress_bar.progress(30)

                messages = get_default_messages_with_developer(
                    "あなたは情報抽出の専門家です。人物情報を正確に抽出してください。"
                )
                messages.append(EasyInputMessageParam(
                    role="user",
                    content=[ResponseInputTextParam(type="input_text", text=user_input)]
                ))

                status_text.text("🤖 人物情報を抽出中...")
                progress_bar.progress(70)

                response = self.client.responses.parse(
                    model=selected_model,
                    input=messages,
                    text_format=PersonInfo,
                )

                status_text.text("✅ 抽出完了!")
                progress_bar.progress(100)

                # 結果表示
                if hasattr(response, 'output_parsed'):
                    person: PersonInfo = response.output_parsed

                    st.success("🎉 人物情報を抽出しました!")

                    # 人物情報表示
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("👤 名前", person.name)
                    with col2:
                        st.metric("🎂 年齢", f"{person.age}歳")

                    # 詳細データ
                    with st.expander("📊 抽出データ", expanded=False):
                        st.json(person.model_dump())

            except Exception as e:
                st.error(f"Simple extraction エラー: {str(e)}")
                logger.error(f"Simple extraction error: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

    @error_handler_ui
    @timer_ui
    def sample_02_011_multiple_extraction(self, selected_model: str):
        """02_011: 基本パターン（複数の構造化データ抽出）"""
        st.subheader("📚 複数エンティティ抽出")

        st.info("""
        **複数エンティティ抽出**では一つのテキストから
        異なる種類の情報を同時に抽出します。
        """)

        # デフォルトテキスト
        default_text = """登場人物:
- 中島美咲 (27歳)
- 田中亮 (34歳)

おすすめ本:
1. 『流浪の月』   著者: 凪良ゆう  (2019年)
2. 『気分上々』   著者: 山田悠介 (2023年)
"""

        user_input, submitted = UIHelper.create_input_form(
            key="multiple_extraction_form",
            input_type="text_area",
            label="人物と書籍情報を入力してください",
            submit_label="📚 抽出",
            value=default_text,
            height=120,
            help="人物情報と書籍情報を含む文章"
        )

        # データ構造表示
        with st.expander("📚 データ構造", expanded=False):
            st.code("""
class PersonInfo(BaseModel):
    name: str = Field(..., description="名前")
    age: int = Field(..., description="年齢")

class BookInfo(BaseModel):
    title: str = Field(..., description="書籍タイトル")
    author: str = Field(..., description="著者")
    year: int = Field(..., description="出版年")

class ExtractedData(BaseModel):
    persons: List[PersonInfo] = Field(default_factory=list, description="人物リスト")
    books: List[BookInfo] = Field(default_factory=list, description="書籍リスト")
            """, language="python")

        if submitted and user_input:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("📚 複数エンティティ抽出を準備中...")
                progress_bar.progress(20)

                messages = get_default_messages_with_developer(
                    "あなたは情報抽出の専門家です。人物情報と書籍情報を同時に抽出してください。"
                )
                messages.append(EasyInputMessageParam(
                    role="user",
                    content=[ResponseInputTextParam(type="input_text", text=user_input)]
                ))

                status_text.text("🤖 複数データを分析中...")
                progress_bar.progress(70)

                response = self.client.responses.parse(
                    model=selected_model,
                    input=messages,
                    text_format=ExtractedData,
                )

                status_text.text("✅ 抽出完了!")
                progress_bar.progress(100)

                # 結果表示
                if hasattr(response, 'output_parsed'):
                    extracted: ExtractedData = response.output_parsed

                    st.success(f"🎉 人物{len(extracted.persons)}名、書籍{len(extracted.books)}冊を抽出しました!")

                    # 2カラム表示
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("👥 人物一覧")
                        if extracted.persons:
                            for i, person in enumerate(extracted.persons, 1):
                                st.write(f"**{i}. {person.name}** ({person.age}歳)")
                        else:
                            st.info("人物情報が見つかりませんでした")

                    with col2:
                        st.subheader("📚 書籍一覧")
                        if extracted.books:
                            for i, book in enumerate(extracted.books, 1):
                                st.write(f"**{i}. {book.title}**")
                                st.write(f"   著者: {book.author} ({book.year}年)")
                        else:
                            st.info("書籍情報が見つかりませんでした")

                    # 統計情報
                    st.subheader("📊 抽出統計")
                    col3, col4 = st.columns(2)
                    with col3:
                        st.metric("👥 人物数", len(extracted.persons))
                    with col4:
                        st.metric("📚 書籍数", len(extracted.books))

                    # JSON出力
                    with st.expander("📊 構造化データ", expanded=False):
                        st.json(extracted.model_dump())

            except Exception as e:
                st.error(f"Multiple extraction エラー: {str(e)}")
                logger.error(f"Multiple extraction error: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

    @error_handler_ui
    @timer_ui
    def sample_02_05_conversation_history(self, selected_model: str):
        """02_05: 会話履歴を持った連続した構造化出力の処理"""
        st.subheader("💬 会話履歴構造化")

        st.info("""
        **会話履歴管理**では連続的な質問応答を構造化して
        蓄積・管理します。
        """)

        # 履歴初期化
        if 'qa_history' not in st.session_state:
            st.session_state.qa_history = []

        # 現在の履歴表示
        if st.session_state.qa_history:
            st.subheader("📜 会話履歴")
            for i, qa in enumerate(st.session_state.qa_history, 1):
                with st.container():
                    st.write(f"**Q{i}**: {qa.question}")
                    st.write(f"**A{i}**: {qa.answer}")
                    st.divider()

        # 新しい質問入力
        user_input, submitted = UIHelper.create_input_form(
            key="conversation_history_form",
            label="質問を入力してください",
            submit_label="❓ 質問",
            placeholder="例: Pythonの用途を教えてください",
            help="新しい質問を追加"
        )

        # QAモデル表示
        with st.expander("💬 QAモデル", expanded=False):
            st.code("""
class QAResponse(BaseModel):
    question: str = Field(..., description="質問")
    answer: str = Field(..., description="回答")
            """, language="python")

        if submitted and user_input:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("💬 質問を処理中...")
                progress_bar.progress(30)

                messages = get_default_messages_with_developer(
                    "あなたは知識豊富なアシスタントです。質問に対して適切な回答を構造化して提供してください。"
                )
                messages.append(EasyInputMessageParam(
                    role="user",
                    content=[ResponseInputTextParam(type="input_text", text=user_input)]
                ))

                status_text.text("🤖 回答を生成中...")
                progress_bar.progress(70)

                response = self.client.responses.parse(
                    model=selected_model,
                    input=messages,
                    text_format=QAResponse,
                )

                status_text.text("✅ 回答完了!")
                progress_bar.progress(100)

                # 結果処理
                if hasattr(response, 'output_parsed'):
                    qa: QAResponse = response.output_parsed

                    # 履歴に追加
                    st.session_state.qa_history.append(qa)

                    st.success("🎉 新しいQ&Aを履歴に追加しました!")

                    # 最新のQ&A表示
                    st.subheader("💡 最新のQ&A")
                    st.write(f"**質問**: {qa.question}")
                    st.write(f"**回答**: {qa.answer}")

                    st.rerun()

            except Exception as e:
                st.error(f"Conversation history エラー: {str(e)}")
                logger.error(f"Conversation history error: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

        # 履歴管理
        if st.session_state.qa_history:
            st.subheader("🔧 履歴管理")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🗑️ 履歴クリア"):
                    st.session_state.qa_history = []
                    st.rerun()

            with col2:
                if st.button("📥 履歴エクスポート"):
                    export_data = {
                        "qa_history" : [qa.model_dump() for qa in st.session_state.qa_history],
                        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    UIHelper.create_download_button(
                        export_data,
                        "qa_history.json",
                        "application/json",
                        "💾 保存"
                    )

            with col3:
                st.metric("Q&A数", len(st.session_state.qa_history))

    def run(self):
        """メインデモ実行"""
        # ページ初期化
        init_page("🔧 Tools & Pydantic Parse 包括デモ", sidebar_title="📋 情報パネル")

        # モデル選択
        selected_model = select_model(self.demo_name)

        # サイドバー設定
        self.setup_sidebar(selected_model)

        # メイン画面
        st.markdown("""
        ## 📖 概要
        OpenAI Responses APIのtools paramとPydantic構造化出力の包括的なデモです。
        基本的な関数呼び出しから複雑なデータ抽出まで様々なパターンを学習できます。
        """)

        # タブでデモを分離
        tabs = st.tabs([
            "🔧 Basic Function",
            "🔄 Multiple Tools",
            "🏗️ Complex Structure",
            "🎯 Enum Types",
            "📝 Text Format",
            "📊 Simple Extract",
            "📚 Multi Extract",
            "💬 History QA"
        ])

        with tabs[0]:
            self.sample_01_01_basic_function_call(selected_model)

        with tabs[1]:
            self.sample_01_021_multiple_tools(selected_model)

        with tabs[2]:
            self.sample_01_03_complex_structure(selected_model)

        with tabs[3]:
            self.sample_01_04_enum_types(selected_model)

        with tabs[4]:
            self.sample_01_05_text_format(selected_model)

        with tabs[5]:
            self.sample_02_01_simple_extraction(selected_model)

        with tabs[6]:
            self.sample_02_011_multiple_extraction(selected_model)

        with tabs[7]:
            self.sample_02_05_conversation_history(selected_model)

        # フッター
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray;'>
        🔧 <b>改修版</b> - 新しいヘルパーモジュールを使用 | 
        📊 Tools & Pydantic Parse 包括デモ |
        🚀 型安全で効率的なAPI活用
        </div>
        """, unsafe_allow_html=True)


# ==================================================
# メイン実行部
# ==================================================
def main():
    """メイン関数"""
    try:
        demo = ToolsPydanticDemo()
        demo.run()
    except Exception as e:
        st.error(f"アプリケーションエラー: {str(e)}")
        logger.error(f"Application error: {e}")

        if config.get("experimental.debug_mode", False):
            st.exception(e)


if __name__ == "__main__":
    main()

# streamlit run a20_02_responses_tools_pydantic_parse.py --server.port=8503