# streamlit run a20_00_responses_skeleton.py --server.port=8501
# 改修版: 新しいヘルパーモジュールを使用した Responses API サンプル
# ==================================================
# 基本機能のデモ: responses.create と responses.parse
# 左ペインに詳細情報を表示するリッチなUI
# ==================================================

import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

import streamlit as st
from pydantic import BaseModel, Field

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


# ==================================================
# 情報パネル表示関数
# ==================================================
class InfoPanelManager:
    """左ペインの情報パネル管理"""

    @staticmethod
    def show_model_info(selected_model: str):
        """モデル情報パネル"""
        with st.sidebar.expander("🤖 モデル情報", expanded=True):
            # 基本情報
            limits = TokenManager.get_model_limits(selected_model)
            pricing = config.get("model_pricing", {}).get(selected_model, {})

            # メトリクス表示
            col1, col2 = st.columns(2)
            with col1:
                st.metric("最大入力", f"{limits['max_tokens']:,}")
                st.metric("最大出力", f"{limits['max_output']:,}")
            with col2:
                if pricing:
                    st.metric("入力料金", f"${pricing.get('input', 0):.5f}/1K")
                    st.metric("出力料金", f"${pricing.get('output', 0):.5f}/1K")

            # モデルカテゴリ
            categories = config.get("models.categories", {})
            model_category = "不明"
            for category, models in categories.items():
                if selected_model in models:
                    model_category = category
                    break

            st.info(f"📂 カテゴリ: {model_category}")

            # モデル特性
            if "reasoning" in model_category:
                st.success("🧠 推論特化モデル - 複雑な問題解決に最適")
            elif "audio" in selected_model:
                st.success("🎵 音声対応モデル - 音声入出力が可能")
            elif "vision" in selected_model or "gpt-4o" in selected_model:
                st.success("👁️ 視覚対応モデル - 画像理解が可能")

    @staticmethod
    def show_session_info():
        """セッション情報パネル"""
        with st.sidebar.expander("📊 セッション情報", expanded=False):
            # セッション統計
            session_data = {}
            for key, value in st.session_state.items():
                if not key.startswith('_'):
                    session_data[key] = str(type(value).__name__)

            st.write("**セッション変数**")
            for key, value_type in list(session_data.items())[:5]:  # 最初の5個のみ表示
                st.write(f"- `{key}`: {value_type}")

            if len(session_data) > 5:
                st.write(f"... 他 {len(session_data) - 5} 個")

            # メモリ使用量（簡易）
            import sys
            session_size = sys.getsizeof(st.session_state)
            st.metric("セッションサイズ", f"{session_size:,} bytes")

    @staticmethod
    def show_performance_info():
        """パフォーマンス情報パネル"""
        metrics = SessionStateManager.get_performance_metrics()
        if not metrics:
            return

        with st.sidebar.expander("⚡ パフォーマンス", expanded=False):
            recent_metrics = metrics[-5:]  # 最近の5回

            if recent_metrics:
                avg_time = sum(m['execution_time'] for m in recent_metrics) / len(recent_metrics)
                max_time = max(m['execution_time'] for m in recent_metrics)
                min_time = min(m['execution_time'] for m in recent_metrics)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("平均", f"{avg_time:.2f}s")
                    st.metric("最大", f"{max_time:.2f}s")
                with col2:
                    st.metric("最小", f"{min_time:.2f}s")
                    st.metric("実行回数", len(metrics))

                # 最新の実行時間
                if metrics:
                    latest = recent_metrics[-1]
                    st.write(f"**最新実行**: {latest['function']} ({latest['execution_time']:.2f}s)")

    @staticmethod
    def show_cost_calculator(selected_model: str):
        """料金計算パネル"""
        with st.sidebar.expander("💰 料金計算", expanded=False):
            pricing = config.get("model_pricing", {}).get(selected_model)
            if not pricing:
                st.warning("料金情報が見つかりません")
                return

            # 入力フィールド
            input_tokens = st.number_input(
                "入力トークン数",
                min_value=0,
                value=1000,
                step=100,
                help="予想される入力トークン数"
            )
            output_tokens = st.number_input(
                "出力トークン数",
                min_value=0,
                value=500,
                step=100,
                help="予想される出力トークン数"
            )

            # 料金計算
            if st.button("💰 料金計算", use_container_width=True):
                cost = TokenManager.estimate_cost(input_tokens, output_tokens, selected_model)

                # 詳細表示
                input_cost = (input_tokens / 1000) * pricing["input"]
                output_cost = (output_tokens / 1000) * pricing["output"]

                st.success(f"**総コスト**: ${cost:.6f}")
                st.write(f"- 入力: ${input_cost:.6f}")
                st.write(f"- 出力: ${output_cost:.6f}")

                # 月間コスト推定
                daily_calls = st.slider("1日の呼び出し回数", 1, 1000, 100)
                monthly_cost = cost * daily_calls * 30
                st.info(f"**月間推定**: ${monthly_cost:.2f}")

    @staticmethod
    def show_debug_panel():
        """デバッグパネル"""
        if not config.get("experimental.debug_mode", False):
            return

        with st.sidebar.expander("🐛 デバッグ情報", expanded=False):
            # 設定情報
            st.write("**アクティブ設定**")
            debug_config = {
                "default_model"         : config.get("models.default"),
                "cache_enabled"         : config.get("cache.enabled"),
                "debug_mode"            : config.get("experimental.debug_mode"),
                "performance_monitoring": config.get("experimental.performance_monitoring")
            }

            for key, value in debug_config.items():
                st.write(f"- {key}: `{value}`")

            # ログレベル
            current_level = config.get("logging.level", "INFO")
            new_level = st.selectbox(
                "ログレベル",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                index=["DEBUG", "INFO", "WARNING", "ERROR"].index(current_level)
            )
            if new_level != current_level:
                config.set("logging.level", new_level)
                logger.setLevel(getattr(logger, new_level))

            # キャッシュ情報
            from helper_api import cache
            st.write(f"**キャッシュ**: {cache.size()} エントリ")
            if st.button("🗑️ キャッシュクリア"):
                cache.clear()
                st.success("キャッシュをクリアしました")


# ==================================================
# デモクラス定義
# ==================================================
class ResponsesSkeletonDemo:
    """Responses API基本デモクラス"""

    def __init__(self):
        self.demo_name = "responses_skeleton"
        self.message_manager = MessageManagerUI(f"messages_{self.demo_name}")
        SessionStateManager.init_session_state()

    def setup_sidebar(self, selected_model: str):
        """左サイドバーの設定"""
        st.sidebar.title("📋 情報パネル")

        # 各情報パネルを表示
        InfoPanelManager.show_model_info(selected_model)
        InfoPanelManager.show_session_info()
        InfoPanelManager.show_performance_info()
        InfoPanelManager.show_cost_calculator(selected_model)
        InfoPanelManager.show_debug_panel()

        # 設定パネル
        UIHelper.show_settings_panel()

    @error_handler_ui
    @timer_ui
    def responses_create_demo(self, selected_model: str):
        """responses.create デモ"""
        st.subheader("🎯 responses.create デモ")

        # 説明
        st.info("""
        **responses.create** は最も基本的なAPI呼び出しです。
        テキスト入力に対してモデルが自然言語で応答します。
        """)

        # 入力フォーム
        user_input, submitted = UIHelper.create_input_form(
            key="create_form",
            label="質問を入力してください",
            submit_label="🚀 送信",
            placeholder="例: OpenAIのResponses APIについて教えて",
            help="何でも気軽に質問してください"
        )

        if submitted and user_input:
            # プログレスバーと状態表示
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # トークン数チェック
                token_count = TokenManager.count_tokens(user_input, selected_model)
                limits = TokenManager.get_model_limits(selected_model)

                if token_count > limits['max_tokens'] * 0.8:
                    st.warning(f"⚠️ 入力が長すぎます ({token_count:,} トークン)")
                    return

                # メッセージ準備
                status_text.text("📝 メッセージを準備中...")
                progress_bar.progress(20)

                messages = self.message_manager.get_default_messages()
                messages.append(EasyInputMessageParam(role="user", content=user_input))

                # API呼び出し
                status_text.text("🤖 AIが回答を生成中...")
                progress_bar.progress(50)

                client = OpenAIClient()
                response = client.create_response(messages, model=selected_model)

                # 結果表示
                status_text.text("✅ 完了!")
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
                logger.error(f"responses.create error: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

    @error_handler_ui
    @timer_ui
    def responses_parse_demo(self, selected_model: str):
        """responses.parse デモ"""
        st.subheader("🎯 responses.parse デモ")

        # 説明
        st.info("""
        **responses.parse** は構造化された出力を生成します。
        Pydanticモデルを使用してJSONスキーマに従った回答を取得できます。
        """)

        # サンプルテキスト
        sample_text = config.get("samples.prompts.event_example",
                                 "私の名前は田中太郎、30歳、東京在住です。私の友人は鈴木健太、28歳、大阪在住です。")

        # 入力フォーム
        user_input, submitted = UIHelper.create_input_form(
            key="parse_form",
            label="人物情報を入力してください",
            submit_label="🔄 構造化",
            value=sample_text,
            help="名前、年齢、住所が含まれるテキストを入力"
        )

        # スキーマ表示
        with st.expander("📋 出力スキーマ", expanded=False):
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
            # プログレスバーと状態表示
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # メッセージ準備
                status_text.text("📝 構造化プロンプトを準備中...")
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

                # API呼び出し
                status_text.text("🔄 構造化データを生成中...")
                progress_bar.progress(70)

                client = OpenAIClient()
                response = client.client.responses.parse(
                    model=selected_model,
                    input=messages,
                    text_format=People
                )

                # 結果処理
                status_text.text("✅ 構造化完了!")
                progress_bar.progress(100)

                # 結果表示
                if hasattr(response, 'output_parsed'):
                    people: People = response.output_parsed

                    st.success(f"🎉 {people.total_count}人の情報を抽出しました!")

                    # 構造化データ表示
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.subheader("👥 抽出された人物情報")
                        for i, person in enumerate(people.users, 1):
                            with st.container():
                                st.write(f"**Person {i}**")
                                person_col1, person_col2, person_col3 = st.columns(3)
                                with person_col1:
                                    st.metric("名前", person.name)
                                with person_col2:
                                    st.metric("年齢", f"{person.age}歳")
                                with person_col3:
                                    st.metric("居住地", person.city)
                                st.divider()

                    with col2:
                        st.subheader("📊 JSON出力")
                        st.json(people.model_dump())

                        # ダウンロードボタン
                        UIHelper.create_download_button(
                            people.model_dump(),
                            "extracted_people.json",
                            "application/json",
                            "📥 JSONダウンロード"
                        )

                # 詳細情報
                with st.expander("📊 API詳細情報", expanded=False):
                    # レスポンス情報の安全な取得
                    response_info = {
                        "model": selected_model,
                        "id"   : getattr(response, 'id', 'N/A')
                    }

                    # usage情報の安全な取得
                    if hasattr(response, 'usage') and response.usage:
                        usage_obj = response.usage
                        if hasattr(usage_obj, 'model_dump'):
                            response_info["usage"] = usage_obj.model_dump()
                        elif hasattr(usage_obj, 'dict'):
                            response_info["usage"] = usage_obj.dict()
                        else:
                            response_info["usage"] = {
                                'prompt_tokens'    : getattr(usage_obj, 'prompt_tokens', 0),
                                'completion_tokens': getattr(usage_obj, 'completion_tokens', 0),
                                'total_tokens'     : getattr(usage_obj, 'total_tokens', 0)
                            }
                    else:
                        response_info["usage"] = {}

                    st.json(response_info)

            except Exception as e:
                st.error(f"構造化エラー: {str(e)}")
                logger.error(f"responses.parse error: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

    def show_message_history(self):
        """メッセージ履歴表示"""
        with st.expander("💬 会話履歴", expanded=False):
            messages = self.message_manager.get_messages()
            if messages:
                UIHelper.display_messages(messages, show_system=True)

                # 履歴操作
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🗑️ 履歴クリア"):
                        self.message_manager.clear_messages()
                        st.rerun()
                with col2:
                    if st.button("📥 履歴エクスポート"):
                        export_data = self.message_manager.export_messages_ui()
                        UIHelper.create_download_button(
                            export_data,
                            "chat_history.json",
                            "application/json",
                            "💾 保存"
                        )
                with col3:
                    st.metric("メッセージ数", len(messages))
            else:
                st.info("会話履歴がありません")

    def run(self):
        """メインデモ実行"""
        # ページ初期化
        init_page("🚀 Responses API 基本デモ", sidebar_title="📋 情報パネル")

        # モデル選択
        selected_model = select_model(self.demo_name)

        # サイドバー設定
        self.setup_sidebar(selected_model)

        # メイン画面
        st.markdown("""
        ## 📖 概要
        このデモでは OpenAI Responses API の基本機能を体験できます：
        - **responses.create**: 自然言語での対話
        - **responses.parse**: 構造化データの生成
        """)

        # タブでデモを分離
        tab1, tab2, tab3 = st.tabs(["💬 Create Demo", "🔄 Parse Demo", "📝 履歴"])

        with tab1:
            self.responses_create_demo(selected_model)

        with tab2:
            self.responses_parse_demo(selected_model)

        with tab3:
            self.show_message_history()

        # フッター
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray;'>
        🔧 <b>改修版</b> - 新しいヘルパーモジュールを使用 | 
        📊 左ペインで詳細情報を確認できます
        </div>
        """, unsafe_allow_html=True)


# ==================================================
# メイン実行部
# ==================================================
def main():
    """メイン関数"""
    try:
        demo = ResponsesSkeletonDemo()
        demo.run()
    except Exception as e:
        st.error(f"アプリケーションエラー: {str(e)}")
        logger.error(f"Application error: {e}")

        if config.get("experimental.debug_mode", False):
            st.exception(e)


if __name__ == "__main__":
    main()

# streamlit run a20_00_responses_skeleton.py --server.port=8501
