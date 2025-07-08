# helper_st.py
# Streamlit UI関連機能
# -----------------------------------------
from functools import wraps
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from abc import ABC, abstractmethod
import json

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

    # ユーティリティ
    sanitize_key,
    format_timestamp,

    # 定数
    config,
    logger
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
            if 'cache' not in st.session_state:
                st.session_state.cache = {}

            # キャッシュの確認
            if cache_key in st.session_state.cache:
                import time
                cached_data = st.session_state.cache[cache_key]
                if time.time() - cached_data['timestamp'] < (ttl or config.get("cache.ttl", 3600)):
                    return cached_data['result']

            # 関数実行とキャッシュ保存
            result = func(*args, **kwargs)
            import time
            st.session_state.cache[cache_key] = {
                'result'   : result,
                'timestamp': time.time()
            }

            # キャッシュサイズ制限
            max_size = config.get("cache.max_size", 100)
            if len(st.session_state.cache) > max_size:
                # 最も古いエントリを削除
                oldest_key = min(st.session_state.cache, key=lambda k: st.session_state.cache[k]['timestamp'])
                del st.session_state.cache[oldest_key]

            return result

        return wrapper

    return decorator


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
            developer_msg = st.session_state[self.session_key][0] if st.session_state[self.session_key][0][
                                                                         'role'] == 'developer' else None
            st.session_state[self.session_key] = st.session_state[self.session_key][-limit:]
            if developer_msg and st.session_state[self.session_key][0]['role'] != 'developer':
                st.session_state[self.session_key].insert(0, developer_msg)

    def get_messages(self) -> List[EasyInputMessageParam]:
        """メッセージ履歴の取得"""
        return st.session_state[self.session_key]

    def clear_messages(self):
        """メッセージ履歴のクリア"""
        st.session_state[self.session_key] = self.get_default_messages()

    def import_messages(self, data: Dict[str, Any]):
        """メッセージ履歴のインポート"""
        if 'messages' in data:
            st.session_state[self.session_key] = data['messages']


# ==================================================
# UI ヘルパー（拡張版）
# ==================================================
class UIHelper:
    """Streamlit UI用のヘルパー関数（拡張版）"""

    @staticmethod
    def init_page(title: str = None, sidebar_title: str = None):
        """ページの初期化"""
        if title is None:
            title = config.get("ui.page_title", "OpenAI API Demo")
        if sidebar_title is None:
            sidebar_title = "メニュー"

        st.set_page_config(
            page_title=title,
            page_icon=config.get("ui.page_icon", "🤖"),
            layout=config.get("ui.layout", "wide")
        )

        st.header(title)
        st.sidebar.title(sidebar_title)

    @staticmethod
    def select_model(key: str = "model_selection", category: str = None) -> str:
        """モデル選択UI（カテゴリ対応）"""
        models = config.get("models.available", ["gpt-4o", "gpt-4o-mini"])
        default_model = config.get("models.default", "gpt-4o-mini")

        # カテゴリでフィルタリング
        if category:
            if category == "reasoning":
                models = [m for m in models if m.startswith("o")]
            elif category == "standard":
                models = [m for m in models if m.startswith("gpt")]
            elif category == "audio":
                models = [m for m in models if "audio" in m]

        default_index = models.index(default_model) if default_model in models else 0

        selected = st.sidebar.selectbox(
            "モデルを選択",
            models,
            index=default_index,
            key=key
        )

        # モデル情報の表示
        with st.sidebar.expander("モデル情報"):
            limits = TokenManager.get_model_limits(selected)
            st.write(f"最大入力: {limits['max_tokens']:,} tokens")
            st.write(f"最大出力: {limits['max_output']:,} tokens")

        return selected

    @staticmethod
    def select_speech_model(key: str = "speech_model_selection", category: str = None) -> str:
        """音声合成モデル選択UI（カテゴリ対応）"""
        all_speech_models = [
            "tts-1", "tts-1-hd",
            "gpt-4o-audio-preview", "gpt-4o-mini-audio-preview",
            "o3-mini", "o4-mini", "o1-mini"
        ]

        default_speech_model = "tts-1"

        # カテゴリでフィルタリング
        if category:
            if category == "tts":
                models = [m for m in all_speech_models if m.startswith("tts")]
            elif category == "audio_chat":
                models = [m for m in all_speech_models if "audio" in m]
            elif category == "reasoning":
                models = [m for m in all_speech_models if m.startswith("o")]
            else:
                models = all_speech_models
        else:
            models = all_speech_models

        default_index = models.index(default_speech_model) if default_speech_model in models else 0

        selected = st.sidebar.selectbox(
            "音声合成モデルを選択",
            models,
            index=default_index,
            key=key
        )

        # モデル情報の表示
        with st.sidebar.expander("音声モデル情報"):
            if selected.startswith("tts"):
                st.write("**TTS専用モデル**")
                if selected == "tts-1":
                    st.write("- 高速・低コスト")
                    st.write("- 音質: 標準")
                elif selected == "tts-1-hd":
                    st.write("- 高音質・低遅延")
                    st.write("- 音質: 高品質")
            elif "audio" in selected:
                st.write("**音声対話モデル**")
                st.write("- テキスト+音声入出力対応")
                st.write("- リアルタイム対話可能")
                limits = TokenManager.get_model_limits(selected)
                st.write(f"最大入力: {limits['max_tokens']:,} tokens")
                st.write(f"最大出力: {limits['max_output']:,} tokens")
            elif selected.startswith("o"):
                st.write("**推論系モデル（音声対応）**")
                st.write("- 高度な推論能力")
                st.write("- 複雑なタスクに対応")
                limits = TokenManager.get_model_limits(selected)
                st.write(f"最大入力: {limits['max_tokens']:,} tokens")
                st.write(f"最大出力: {limits['max_output']:,} tokens")

        return selected

    @staticmethod
    def select_whisper_model(key: str = "whisper_model_selection", category: str = None) -> str:
        """音声認識/翻訳モデル選択UI（カテゴリ対応）"""
        all_whisper_models = [
            "whisper-1",
            "gpt-4o-transcribe", "gpt-4o-mini-transcribe",
            "gpt-4o-audio-preview", "gpt-4o-mini-audio-preview"
        ]

        default_whisper_model = "whisper-1"

        # カテゴリでフィルタリング
        if category:
            if category == "whisper":
                models = [m for m in all_whisper_models if "whisper" in m]
            elif category == "transcribe":
                models = [m for m in all_whisper_models if "transcribe" in m]
            elif category == "audio_chat":
                models = [m for m in all_whisper_models if "audio-preview" in m]
            elif category == "gpt":
                models = [m for m in all_whisper_models if m.startswith("gpt")]
            else:
                models = all_whisper_models
        else:
            models = all_whisper_models

        default_index = models.index(default_whisper_model) if default_whisper_model in models else 0

        selected = st.sidebar.selectbox(
            "音声認識/翻訳モデルを選択",
            models,
            index=default_index,
            key=key
        )

        # モデル情報の表示
        with st.sidebar.expander("音声認識モデル情報"):
            if selected == "whisper-1":
                st.write("**Whisper専用モデル**")
                st.write("- 多言語対応")
                st.write("- 転写・翻訳対応")
                st.write("- ファイルサイズ: 最大25MB")
                st.write("- 対応形式: mp3, mp4, wav, webm, m4a, flac, etc.")
            elif "transcribe" in selected:
                st.write("**GPT系転写モデル**")
                st.write("- 高精度転写")
                st.write("- コンテキスト理解")
                if "mini" in selected:
                    st.write("- 高速・低コスト版")
                else:
                    st.write("- 高性能版")
            elif "audio-preview" in selected:
                st.write("**音声対話モデル（STT機能）**")
                st.write("- リアルタイム音声処理")
                st.write("- テキスト+音声入出力")
                limits = TokenManager.get_model_limits(selected)
                st.write(f"最大入力: {limits['max_tokens']:,} tokens")
                st.write(f"最大出力: {limits['max_output']:,} tokens")

            st.write("---")
            st.write("**対応言語**: 日本語、英語、その他多数")

        return selected

    @staticmethod
    def create_form(key: str, submit_label: str = "送信") -> Tuple[Any, bool]:
        """フォームの作成"""
        form = st.form(key=key)
        submitted = form.form_submit_button(submit_label)
        return form, submitted

    @staticmethod
    def display_messages(messages: List[EasyInputMessageParam], show_system: bool = False):
        """メッセージ履歴の表示（改良版）"""
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                with st.chat_message("user"):
                    if isinstance(content, list):
                        # マルチモーダルコンテンツの処理
                        for item in content:
                            if item.get("type") == "input_text":
                                st.markdown(item.get("text", ""))
                            elif item.get("type") == "input_image":
                                st.image(item.get("image_url", ""))
                    else:
                        st.markdown(content)
            elif role == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(content)
            elif (role == "developer" or role == "system") and show_system:
                with st.expander(f"{role.capitalize()} Message", expanded=False):
                    st.markdown(f"*{content}*")

    @staticmethod
    def show_token_info(text: str, model: str = None):
        """トークン情報の表示（拡張版）"""
        token_count = TokenManager.count_tokens(text, model)
        limits = TokenManager.get_model_limits(model)

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("トークン数", f"{token_count:,}")
        with col2:
            usage_percent = (token_count / limits['max_tokens']) * 100
            st.metric("使用率", f"{usage_percent:.1f}%")

        # コスト推定（仮定: 出力は入力の50%）
        estimated_output = token_count // 2
        cost = TokenManager.estimate_cost(token_count, estimated_output, model)
        st.sidebar.metric("推定コスト", f"${cost:.4f}")

        # プログレスバー
        st.sidebar.progress(min(usage_percent / 100, 1.0))

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
                    st.metric(label, value.get('value'), value.get('delta'))
                else:
                    st.metric(label, value)

    @staticmethod
    def create_download_button(data: Any, filename: str, mime_type: str = "text/plain", label: str = "ダウンロード"):
        """ダウンロードボタンの作成"""
        if isinstance(data, dict):
            data = json.dumps(data, ensure_ascii=False, indent=2)
        elif isinstance(data, list):
            data = json.dumps(data, ensure_ascii=False, indent=2)

        st.download_button(
            label=label,
            data=data,
            file_name=filename,
            mime=mime_type
        )


# ==================================================
# レスポンス処理（UI拡張）
# ==================================================
class ResponseProcessorUI(ResponseProcessor):
    """API レスポンスの処理（UI拡張）"""

    @staticmethod
    def display_response(response: Response, show_details: bool = True):
        """レスポンスの表示（改良版）"""
        texts = ResponseProcessor.extract_text(response)

        if texts:
            for i, text in enumerate(texts, 1):
                if len(texts) > 1:
                    st.subheader(f"回答 {i}")
                st.write(text)
        else:
            st.warning("テキストが見つかりませんでした")

        # 詳細情報の表示
        if show_details:
            with st.expander("詳細情報"):
                formatted = ResponseProcessor.format_response(response)

                # 使用状況の表示
                if 'usage' in formatted and formatted['usage']:
                    usage = formatted['usage']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("入力トークン", usage.get('prompt_tokens', 0))
                    with col2:
                        st.metric("出力トークン", usage.get('completion_tokens', 0))
                    with col3:
                        st.metric("合計トークン", usage.get('total_tokens', 0))

                # JSON形式での表示
                st.json(formatted)


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

    @abstractmethod
    def run(self):
        """デモの実行（サブクラスで実装）"""
        pass

    def setup_ui(self):
        """共通UI設定"""
        st.subheader(self.title)

        # モデル選択
        self.model = UIHelper.select_model(f"model_{self.key_prefix}")

        # メッセージ履歴のクリア
        if st.sidebar.button("履歴クリア", key=f"clear_{self.key_prefix}"):
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
        from openai import OpenAI
        client = OpenAI()

        # デフォルトパラメータ
        params = {
            "model": self.model,
            "input": messages,
        }
        params.update(kwargs)

        # API呼び出し
        response = client.responses.create(**params)
        return response


# ==================================================
# 後方互換性のための関数
# ==================================================
def init_page(title: str):
    """後方互換性のための関数"""
    UIHelper.init_page(title)


def init_messages(demo_name: str = ""):
    """後方互換性のための関数"""
    manager = MessageManagerUI(f"messages_{sanitize_key(demo_name)}")

    if st.sidebar.button("会話履歴のクリア", key=f"clear_{sanitize_key(demo_name)}"):
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