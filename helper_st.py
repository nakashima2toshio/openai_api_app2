# helper_st.py
# Streamlit UI関連機能
# -----------------------------------------
from functools import wraps
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from abc import ABC, abstractmethod
import json
import time
import traceback  # ← 追加

import streamlit as st

# helper_api.pyから必要な機能をインポート
from helper_api import (
    # 型定義
    RoleType,
    EasyInputMessageParam,
    ResponseInputTextParam,
    ResponseInputImageParam,
    Response,

    # クラス
    ConfigManager,
    MessageManager,
    TokenManager,
    ResponseProcessor,
    OpenAIClient,

    # ユーティリティ
    sanitize_key,
    format_timestamp,
    save_json_file,

    # グローバル
    config,
    logger,
    cache,
)


# ==================================================
# デコレータ（Streamlit UI用）
# ==================================================
def error_handler_ui(func):
    """エラーハンドリングデコレータ（Streamlit UI用）"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            error_msg = config.get("error_messages.general_error", f"エラーが発生しました: {str(e)}")
            st.error(error_msg)
            if config.get("experimental.debug_mode", False):
                st.exception(e)
            return None

    return wrapper


def timer_ui(func):
    """実行時間計測デコレータ（Streamlit UI用）"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        logger.info(f"{func.__name__} took {execution_time:.2f} seconds")

        # パフォーマンスモニタリングが有効な場合
        if config.get("experimental.performance_monitoring", True):
            if 'performance_metrics' not in st.session_state:
                st.session_state.performance_metrics = []
            st.session_state.performance_metrics.append({
                'function'      : func.__name__,
                'execution_time': execution_time,
                'timestamp'     : datetime.now()
            })

        return result

    return wrapper


def cache_result_ui(ttl: int = None):
    """結果をキャッシュするデコレータ（Streamlit session_state用）"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not config.get("cache.enabled", True):
                return func(*args, **kwargs)

            # キャッシュキーの生成
            import hashlib
            cache_key = f"{func.__name__}_{hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()}"

            # セッションステートにキャッシュ領域を確保
            if 'ui_cache' not in st.session_state:
                st.session_state.ui_cache = {}

            # キャッシュの確認
            if cache_key in st.session_state.ui_cache:
                cached_data = st.session_state.ui_cache[cache_key]
                if time.time() - cached_data['timestamp'] < (ttl or config.get("cache.ttl", 3600)):
                    return cached_data['result']

            # 関数実行とキャッシュ保存
            result = func(*args, **kwargs)
            st.session_state.ui_cache[cache_key] = {
                'result'   : result,
                'timestamp': time.time()
            }

            # キャッシュサイズ制限
            max_size = config.get("cache.max_size", 100)
            if len(st.session_state.ui_cache) > max_size:
                # 最も古いエントリを削除
                oldest_key = min(st.session_state.ui_cache,
                                 key=lambda k: st.session_state.ui_cache[k]['timestamp'])
                del st.session_state.ui_cache[oldest_key]

            return result

        return wrapper

    return decorator


# ==================================================
# セッション状態管理
# ==================================================
class SessionStateManager:
    """Streamlit セッション状態の管理"""

    @staticmethod
    def init_session_state():
        """セッション状態の初期化"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.ui_cache = {}
            st.session_state.performance_metrics = []
            st.session_state.user_preferences = {}

    @staticmethod
    def get_user_preference(key: str, default: Any = None) -> Any:
        """ユーザー設定の取得"""
        return st.session_state.get('user_preferences', {}).get(key, default)

    @staticmethod
    def set_user_preference(key: str, value: Any):
        """ユーザー設定の保存"""
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {}
        st.session_state.user_preferences[key] = value

    @staticmethod
    def clear_cache():
        """UIキャッシュのクリア"""
        st.session_state.ui_cache = {}
        cache.clear()

    @staticmethod
    def get_performance_metrics() -> List[Dict[str, Any]]:
        """パフォーマンスメトリクスの取得"""
        return st.session_state.get('performance_metrics', [])


# ==================================================
# メッセージ管理（Streamlit用）
# ==================================================
class MessageManagerUI(MessageManager):
    """メッセージ履歴の管理（Streamlit UI用）"""

    def __init__(self, session_key: str = "message_history"):
        self.session_key = session_key
        self._initialize_messages()

    def _initialize_messages(self):
        """メッセージ履歴の初期化"""
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = self.get_default_messages()

    def add_message(self, role: RoleType, content: str):
        """メッセージの追加"""
        valid_roles: List[RoleType] = ["user", "assistant", "system", "developer"]
        if role not in valid_roles:
            raise ValueError(f"Invalid role: {role}. Must be one of {valid_roles}")

        st.session_state[self.session_key].append(
            EasyInputMessageParam(role=role, content=content)
        )

        # メッセージ数制限
        limit = config.get("ui.message_display_limit", 50)
        if len(st.session_state[self.session_key]) > limit:
            # 最初のdeveloperメッセージは保持
            messages = st.session_state[self.session_key]
            developer_msg = messages[0] if messages and messages[0].get('role') == 'developer' else None
            st.session_state[self.session_key] = messages[-limit:]
            if developer_msg and st.session_state[self.session_key][0].get('role') != 'developer':
                st.session_state[self.session_key].insert(0, developer_msg)

    def get_messages(self) -> List[EasyInputMessageParam]:
        """メッセージ履歴の取得"""
        return st.session_state.get(self.session_key, [])

    def clear_messages(self):
        """メッセージ履歴のクリア"""
        st.session_state[self.session_key] = self.get_default_messages()

    def import_messages(self, data: Dict[str, Any]):
        """メッセージ履歴のインポート"""
        if 'messages' in data:
            st.session_state[self.session_key] = data['messages']

    def export_messages_ui(self) -> str:
        """メッセージ履歴のエクスポート（UI用）"""
        data = self.export_messages()
        return json.dumps(data, ensure_ascii=False, indent=2)


# ==================================================
# UI ヘルパー（拡張版）
# ==================================================
class UIHelper:
    """Streamlit UI用のヘルパー関数（拡張版）"""

    @staticmethod
    def init_page(title: str = None, sidebar_title: str = None, **kwargs):
        """ページの初期化"""
        # セッション状態の初期化
        SessionStateManager.init_session_state()

        if title is None:
            title = config.get("ui.page_title", "OpenAI API Demo")
        if sidebar_title is None:
            sidebar_title = "メニュー"

        # Streamlit設定
        page_config = {
            "page_title"           : title,
            "page_icon"            : config.get("ui.page_icon", "🤖"),
            "layout"               : config.get("ui.layout", "wide"),
            "initial_sidebar_state": "expanded"
        }
        page_config.update(kwargs)

        # 既に設定済みかチェック
        try:
            st.set_page_config(**page_config)
        except st.errors.StreamlitAPIException:
            # 既に設定済みの場合は無視
            pass

        st.header(title)
        st.sidebar.title(sidebar_title)

        # デバッグ情報の表示（デバッグモード時）
        if config.get("experimental.debug_mode", False):
            UIHelper._show_debug_info()

    @staticmethod
    def _show_debug_info():
        """デバッグ情報の表示"""
        with st.sidebar.expander("🐛 デバッグ情報", expanded=False):
            st.write("**設定情報**")
            st.json(config._config)

            st.write("**セッション状態**")
            st.json({k: str(v)[:100] for k, v in st.session_state.items()})

            st.write("**パフォーマンス**")
            metrics = SessionStateManager.get_performance_metrics()
            if metrics:
                avg_time = sum(m['execution_time'] for m in metrics[-10:]) / min(len(metrics), 10)
                st.metric("平均実行時間（直近10回）", f"{avg_time:.2f}s")

    @staticmethod
    def select_model(key: str = "model_selection", category: str = None, show_info: bool = True) -> str:
        """モデル選択UI（カテゴリ対応）"""
        models = config.get("models.available", ["gpt-4o", "gpt-4o-mini"])
        default_model = config.get("models.default", "gpt-4o-mini")

        # カテゴリでフィルタリング
        if category:
            if category == "reasoning":
                models = [m for m in models if m.startswith("o")]
                st.sidebar.caption("🧠 推論特化モデル")
            elif category == "standard":
                models = [m for m in models if m.startswith("gpt")]
                st.sidebar.caption("💬 標準対話モデル")
            elif category == "audio":
                models = [m for m in models if "audio" in m]
                st.sidebar.caption("🎵 音声対応モデル")

        default_index = models.index(default_model) if default_model in models else 0

        selected = st.sidebar.selectbox(
            "モデルを選択",
            models,
            index=default_index,
            key=key,
            help="利用するOpenAIモデルを選択してください"
        )

        # モデル情報の表示
        if show_info:
            with st.sidebar.expander("📊 モデル情報", expanded=False):
                limits = TokenManager.get_model_limits(selected)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("最大入力", f"{limits['max_tokens']:,}")
                with col2:
                    st.metric("最大出力", f"{limits['max_output']:,}")

                # 料金情報
                pricing = config.get("model_pricing", {}).get(selected)
                if pricing:
                    st.write("**料金（1000トークンあたり）**")
                    st.write(f"- 入力: ${pricing.get('input', 0):.5f}")
                    st.write(f"- 出力: ${pricing.get('output', 0):.5f}")

        # ユーザー設定として保存
        SessionStateManager.set_user_preference("selected_model", selected)

        return selected

    @staticmethod
    def create_input_form(
            key: str,
            input_type: str = "text_area",
            label: str = "入力してください",
            submit_label: str = "送信",
            **kwargs
    ) -> Tuple[str, bool]:
        """入力フォームの作成"""

        with st.form(key=key):
            if input_type == "text_area":
                user_input = st.text_area(
                    label,
                    height=kwargs.get("height", config.get("ui.text_area_height", 75)),
                    **{k: v for k, v in kwargs.items() if k != "height"}
                )
            elif input_type == "text_input":
                user_input = st.text_input(label, **kwargs)
            elif input_type == "file_uploader":
                user_input = st.file_uploader(label, **kwargs)
            else:
                raise ValueError(f"Unsupported input_type: {input_type}")

            # 送信ボタンの設定
            col1, col2 = st.columns([3, 1])
            with col2:
                submitted = st.form_submit_button(submit_label, use_container_width=True)

            return user_input, submitted

    @staticmethod
    def display_messages(messages: List[EasyInputMessageParam], show_system: bool = False):
        """メッセージ履歴の表示（改良版）"""
        if not messages:
            st.info("メッセージがありません")
            return

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                with st.chat_message("user", avatar="👤"):
                    if isinstance(content, list):
                        # マルチモーダルコンテンツの処理
                        for item in content:
                            if item.get("type") == "input_text":
                                st.markdown(item.get("text", ""))
                            elif item.get("type") == "input_image":
                                image_url = item.get("image_url", "")
                                if image_url:
                                    st.image(image_url, caption="アップロード画像")
                    else:
                        st.markdown(content)

            elif role == "assistant":
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(content)

            elif (role == "developer" or role == "system") and show_system:
                with st.expander(f"🔧 {role.capitalize()} Message", expanded=False):
                    st.markdown(f"*{content}*")

    @staticmethod
    def show_token_info(text: str, model: str = None, position: str = "sidebar"):
        """トークン情報の表示（拡張版）"""
        if not text:
            return

        token_count = TokenManager.count_tokens(text, model)
        limits = TokenManager.get_model_limits(model)

        # 表示位置の選択
        container = st.sidebar if position == "sidebar" else st

        with container.container():
            col1, col2 = st.columns(2)
            with col1:
                st.metric("トークン数", f"{token_count:,}")
            with col2:
                usage_percent = (token_count / limits['max_tokens']) * 100
                st.metric("使用率", f"{usage_percent:.1f}%")

            # コスト推定（仮定: 出力は入力の50%）
            estimated_output = token_count // 2
            cost = TokenManager.estimate_cost(token_count, estimated_output, model)
            st.metric("推定コスト", f"${cost:.6f}")

            # プログレスバー
            progress_value = min(usage_percent / 100, 1.0)
            st.progress(progress_value)

            # 警告表示
            if usage_percent > 90:
                st.warning("⚠️ トークン使用率が高いです")
            elif usage_percent > 70:
                st.info("ℹ️ トークン使用率が高めです")

    @staticmethod
    def create_tabs(tab_names: List[str], key: str = "tabs") -> List[Any]:
        """タブの作成"""
        return st.tabs(tab_names)

    @staticmethod
    def create_columns(spec: List[Union[int, float]], gap: str = "medium") -> List[Any]:
        """カラムの作成"""
        return st.columns(spec, gap=gap)

    @staticmethod
    def show_metrics(metrics: Dict[str, Any], columns: int = 3):
        """メトリクスの表示"""
        cols = st.columns(columns)
        for i, (label, value) in enumerate(metrics.items()):
            with cols[i % columns]:
                if isinstance(value, dict):
                    st.metric(
                        label,
                        value.get('value'),
                        delta=value.get('delta'),
                        help=value.get('help')
                    )
                else:
                    st.metric(label, value)

    @staticmethod
    def create_download_button(
            data: Any,
            filename: str,
            mime_type: str = "text/plain",
            label: str = "ダウンロード",
            help: str = None
    ):
        """ダウンロードボタンの作成"""
        if isinstance(data, (dict, list)):
            data = json.dumps(data, ensure_ascii=False, indent=2)
            if mime_type == "text/plain":
                mime_type = "application/json"

        st.download_button(
            label=label,
            data=data,
            file_name=filename,
            mime=mime_type,
            help=help or f"{filename}をダウンロードします"
        )

    @staticmethod
    def show_settings_panel():
        """設定パネルの表示"""
        with st.sidebar.expander("⚙️ 設定", expanded=False):
            # テーマ設定
            theme = st.selectbox(
                "テーマ",
                ["auto", "light", "dark"],
                index=0,
                help="アプリケーションのテーマを選択"
            )
            SessionStateManager.set_user_preference("theme", theme)

            # デバッグモード
            debug_mode = st.checkbox(
                "デバッグモード",
                value=config.get("experimental.debug_mode", False),
                help="詳細なデバッグ情報を表示"
            )
            config.set("experimental.debug_mode", debug_mode)

            # パフォーマンス監視
            perf_monitoring = st.checkbox(
                "パフォーマンス監視",
                value=config.get("experimental.performance_monitoring", True),
                help="関数の実行時間を記録"
            )
            config.set("experimental.performance_monitoring", perf_monitoring)

            # キャッシュ管理
            st.write("**キャッシュ管理**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("キャッシュクリア", help="全キャッシュをクリア"):
                    SessionStateManager.clear_cache()
                    st.success("キャッシュをクリアしました")
            with col2:
                cache_size = cache.size()
                st.metric("キャッシュ数", cache_size)

    @staticmethod
    def show_performance_panel():
        """パフォーマンスパネルの表示"""
        metrics = SessionStateManager.get_performance_metrics()
        if not metrics:
            st.info("パフォーマンスデータがありません")
            return

        with st.expander("📈 パフォーマンス情報", expanded=False):
            # 最近の実行時間
            recent_metrics = metrics[-10:]
            avg_time = sum(m['execution_time'] for m in recent_metrics) / len(recent_metrics)
            max_time = max(m['execution_time'] for m in recent_metrics)
            min_time = min(m['execution_time'] for m in recent_metrics)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("平均実行時間", f"{avg_time:.2f}s")
            with col2:
                st.metric("最大実行時間", f"{max_time:.2f}s")
            with col3:
                st.metric("最小実行時間", f"{min_time:.2f}s")

            # 実行時間の推移
            if len(metrics) > 1:
                import pandas as pd
                df = pd.DataFrame(metrics)
                st.line_chart(df.set_index('timestamp')['execution_time'])


# ==================================================
# レスポンス処理（UI拡張）
# ==================================================
class ResponseProcessorUI(ResponseProcessor):
    """API レスポンスの処理（UI拡張）"""

    @staticmethod
    def display_response(response: Response, show_details: bool = True, show_raw: bool = False):
        """レスポンスの表示（改良版）"""
        texts = ResponseProcessor.extract_text(response)

        if texts:
            for i, text in enumerate(texts, 1):
                if len(texts) > 1:
                    st.subheader(f"🤖 回答 {i}")
                else:
                    st.subheader("🤖 回答")

                # コピーボタン付きで表示
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(text)
                with col2:
                    if st.button("📋", key=f"copy_{i}", help="回答をコピー"):
                        st.write("📋 コピーしました")  # 実際のコピー機能はブラウザ制限により困難
        else:
            st.warning("⚠️ テキストが見つかりませんでした")

        # 詳細情報の表示
        if show_details:
            with st.expander("📊 詳細情報", expanded=False):
                formatted = ResponseProcessor.format_response(response)

                # 使用状況の表示（安全なアクセス）
                usage_data = formatted.get('usage', {})
                if usage_data and isinstance(usage_data, dict):
                    st.write("**トークン使用量**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        prompt_tokens = usage_data.get('prompt_tokens', 0)
                        st.metric("入力", prompt_tokens)
                    with col2:
                        completion_tokens = usage_data.get('completion_tokens', 0)
                        st.metric("出力", completion_tokens)
                    with col3:
                        total_tokens = usage_data.get('total_tokens', 0)
                        st.metric("合計", total_tokens)

                    # コスト計算
                    model = formatted.get('model')
                    if model and (prompt_tokens > 0 or completion_tokens > 0):
                        cost = TokenManager.estimate_cost(
                            prompt_tokens,
                            completion_tokens,
                            model
                        )
                        st.metric("推定コスト", f"${cost:.6f}")
                elif hasattr(response, 'usage') and response.usage:
                    # usageオブジェクトから直接取得（フォールバック）
                    st.write("**トークン使用量**")
                    usage_obj = response.usage
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        prompt_tokens = getattr(usage_obj, 'prompt_tokens', 0)
                        st.metric("入力", prompt_tokens)
                    with col2:
                        completion_tokens = getattr(usage_obj, 'completion_tokens', 0)
                        st.metric("出力", completion_tokens)
                    with col3:
                        total_tokens = getattr(usage_obj, 'total_tokens', 0)
                        st.metric("合計", total_tokens)

                # レスポンス情報
                st.write("**レスポンス情報**")
                info_data = {
                    "ID"      : formatted.get('id', 'N/A'),
                    "モデル"  : formatted.get('model', 'N/A'),
                    "作成日時": formatted.get('created_at', 'N/A')
                }

                for key, value in info_data.items():
                    st.write(f"- **{key}**: {value}")

                # Raw JSON表示
                if show_raw:
                    st.write("**Raw JSON**")
                    st.json(formatted)

                # ダウンロードボタン
                UIHelper.create_download_button(
                    formatted,
                    f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    "📥 JSONダウンロード"
                )


# ==================================================
# デモ基底クラス
# ==================================================
class DemoBase(ABC):
    """デモの基底クラス"""

    def __init__(self, demo_name: str, title: str = None):
        self.demo_name = demo_name
        self.title = title or demo_name
        self.key_prefix = sanitize_key(demo_name)
        self.message_manager = MessageManagerUI(f"messages_{self.key_prefix}")

        # セッション状態の初期化
        SessionStateManager.init_session_state()

    @abstractmethod
    def run(self):
        """デモの実行（サブクラスで実装）"""
        pass

    def setup_ui(self):
        """共通UI設定"""
        st.subheader(self.title)

        # モデル選択
        self.model = UIHelper.select_model(f"model_{self.key_prefix}")

        # 設定パネル
        UIHelper.show_settings_panel()

        # メッセージ履歴のクリア
        if st.sidebar.button("🗑️ 履歴クリア", key=f"clear_{self.key_prefix}"):
            self.message_manager.clear_messages()
            st.rerun()

    def display_messages(self):
        """メッセージの表示"""
        messages = self.message_manager.get_messages()
        UIHelper.display_messages(messages)

    def add_user_message(self, content: str):
        """ユーザーメッセージの追加"""
        self.message_manager.add_message("user", content)

    def add_assistant_message(self, content: str):
        """アシスタントメッセージの追加"""
        self.message_manager.add_message("assistant", content)

    @error_handler_ui
    @timer_ui
    def call_api(self, messages: List[EasyInputMessageParam], **kwargs) -> Response:
        """API呼び出し（共通処理）"""
        client = OpenAIClient()

        # デフォルトパラメータ
        params = {
            "model": self.model,
            "input": messages,
        }
        params.update(kwargs)

        # API呼び出し
        response = client.create_response(**params)
        return response


# ==================================================
# 後方互換性のための関数
# ==================================================
def init_page(title: str, **kwargs):
    """後方互換性のための関数"""
    UIHelper.init_page(title, **kwargs)


def init_messages(demo_name: str = ""):
    """後方互換性のための関数"""
    manager = MessageManagerUI(f"messages_{sanitize_key(demo_name)}")

    if st.sidebar.button("🗑️ 会話履歴のクリア", key=f"clear_{sanitize_key(demo_name)}"):
        manager.clear_messages()


def select_model(demo_name: str = "") -> str:
    """後方互換性のための関数"""
    return UIHelper.select_model(f"model_{sanitize_key(demo_name)}")


def get_default_messages() -> List[EasyInputMessageParam]:
    """後方互換性のための関数"""
    manager = MessageManagerUI()
    return manager.get_default_messages()


def extract_text_from_response(response: Response) -> List[str]:
    """後方互換性のための関数"""
    return ResponseProcessor.extract_text(response)


def append_user_message(append_text: str, image_url: Optional[str] = None) -> List[EasyInputMessageParam]:
    """後方互換性のための関数"""
    messages = get_default_messages()
    if image_url:
        content = [
            ResponseInputTextParam(type="input_text", text=append_text),
            ResponseInputImageParam(type="input_image", image_url=image_url, detail="auto")
        ]
        messages.append(EasyInputMessageParam(role="user", content=content))
    else:
        messages.append(EasyInputMessageParam(role="user", content=append_text))
    return messages


# ==================================================
# エクスポート
# ==================================================
__all__ = [
    # クラス
    'UIHelper',
    'MessageManagerUI',
    'ResponseProcessorUI',
    'DemoBase',
    'SessionStateManager',

    # デコレータ
    'error_handler_ui',
    'timer_ui',
    'cache_result_ui',

    # 後方互換性
    'init_page',
    'init_messages',
    'select_model',
    'get_default_messages',
    'extract_text_from_response',
    'append_user_message',
]