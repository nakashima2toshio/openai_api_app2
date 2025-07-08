# helper_api.py
# OpenAI API関連とコア機能
# -----------------------------------------
import re
import time
import json
import logging
import yaml
from typing import List, Dict, Any, Optional, Union, Tuple, Literal, Callable
from pathlib import Path
from dataclasses import dataclass
from functools import wraps
from datetime import datetime
from abc import ABC, abstractmethod
import hashlib

import tiktoken
from openai import OpenAI

# -----------------------------------------------------
# OpenAI API型定義
# -----------------------------------------------------
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputTextParam,
    ResponseInputImageParam,
    Response
)
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# 定数定義
# --------------------------------------------------
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

# Role型の定義
RoleType = Literal["user", "assistant", "system", "developer"]


# ==================================================
# 設定管理
# ==================================================
class ConfigManager:
    """設定ファイルの管理"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._cache = {}

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"設定ファイルの読み込みに失敗: {e}")
                return self._get_default_config()
        else:
            logger.warning(f"設定ファイルが見つかりません: {self.config_path}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "models": {
                "default"  : "gpt-4o-mini",
                "available": ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"]
            },
            "api"   : {
                "timeout"    : 30,
                "max_retries": 3
            },
            "ui"    : {
                "page_title": "OpenAI API Demo",
                "layout"    : "wide"
            },
            "cache" : {
                "enabled" : True,
                "ttl"     : 3600,
                "max_size": 100
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """設定値の取得（キャッシュ付き）"""
        if key in self._cache:
            return self._cache[key]

        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                value = default
                break

        result = value if value is not None else default
        self._cache[key] = result
        return result

    def reload(self):
        """設定の再読み込み"""
        self._config = self._load_config()
        self._cache.clear()


# グローバル設定インスタンス
config = ConfigManager()

# ==================================================
# メモリベースキャッシュ
# ==================================================
_cache_storage = {}


def clear_cache():
    """キャッシュクリア"""
    global _cache_storage
    _cache_storage.clear()


# ==================================================
# デコレータ（API用）
# ==================================================
def error_handler(func):
    """エラーハンドリングデコレータ（API用）"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            # API用では例外を再発生させる
            raise

    return wrapper


def timer(func):
    """実行時間計測デコレータ（API用）"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} took {execution_time:.2f} seconds")
        return result

    return wrapper


def cache_result(ttl: int = None):
    """結果をキャッシュするデコレータ（メモリベース）"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not config.get("cache.enabled", True):
                return func(*args, **kwargs)

            # キャッシュキーの生成
            cache_key = f"{func.__name__}_{hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()}"

            # キャッシュの確認
            if cache_key in _cache_storage:
                cached_data = _cache_storage[cache_key]
                if time.time() - cached_data['timestamp'] < (ttl or config.get("cache.ttl", 3600)):
                    return cached_data['result']

            # 関数実行とキャッシュ保存
            result = func(*args, **kwargs)
            _cache_storage[cache_key] = {
                'result'   : result,
                'timestamp': time.time()
            }

            # キャッシュサイズ制限
            max_size = config.get("cache.max_size", 100)
            if len(_cache_storage) > max_size:
                # 最も古いエントリを削除
                oldest_key = min(_cache_storage, key=lambda k: _cache_storage[k]['timestamp'])
                del _cache_storage[oldest_key]

            return result

        return wrapper

    return decorator


# ==================================================
# メッセージ管理
# ==================================================
class MessageManager:
    """メッセージ履歴の管理（API用）"""

    def __init__(self, messages: List[EasyInputMessageParam] = None):
        self._messages = messages or self.get_default_messages()

    @staticmethod
    def get_default_messages() -> List[EasyInputMessageParam]:
        """デフォルトメッセージの取得"""
        messages = config.get("default_messages", {})
        return [
            EasyInputMessageParam(
                role="developer",
                content=messages.get("developer", developer_text)
            ),
            EasyInputMessageParam(
                role="user",
                content=messages.get("user", user_text)
            ),
            EasyInputMessageParam(
                role="assistant",
                content=messages.get("assistant", assistant_text)
            ),
        ]

    def add_message(self, role: RoleType, content: str):
        """メッセージの追加"""
        valid_roles: List[RoleType] = ["user", "assistant", "system", "developer"]
        if role not in valid_roles:
            raise ValueError(f"Invalid role: {role}. Must be one of {valid_roles}")

        self._messages.append(
            EasyInputMessageParam(role=role, content=content)
        )

        # メッセージ数制限
        limit = config.get("api.message_limit", 50)
        if len(self._messages) > limit:
            # 最初のdeveloperメッセージは保持
            developer_msg = self._messages[0] if self._messages[0]['role'] == 'developer' else None
            self._messages = self._messages[-limit:]
            if developer_msg and self._messages[0]['role'] != 'developer':
                self._messages.insert(0, developer_msg)

    def get_messages(self) -> List[EasyInputMessageParam]:
        """メッセージ履歴の取得"""
        return self._messages.copy()

    def clear_messages(self):
        """メッセージ履歴のクリア"""
        self._messages = self.get_default_messages()

    def export_messages(self) -> Dict[str, Any]:
        """メッセージ履歴のエクスポート"""
        return {
            'messages'   : self.get_messages(),
            'exported_at': datetime.now().isoformat()
        }

    def import_messages(self, data: Dict[str, Any]):
        """メッセージ履歴のインポート"""
        if 'messages' in data:
            self._messages = data['messages']


# ==================================================
# トークン管理
# ==================================================
class TokenManager:
    """トークン数の管理（新モデル対応）"""

    # モデル別のエンコーディング対応表
    MODEL_ENCODINGS = {
        "gpt-4o"                   : "cl100k_base",
        "gpt-4o-mini"              : "cl100k_base",
        "gpt-4o-audio-preview"     : "cl100k_base",
        "gpt-4o-mini-audio-preview": "cl100k_base",
        "gpt-4.1"                  : "cl100k_base",
        "gpt-4.1-mini"             : "cl100k_base",
        "o1"                       : "cl100k_base",
        "o1-mini"                  : "cl100k_base",
        "o3"                       : "cl100k_base",
        "o3-mini"                  : "cl100k_base",
        "o4"                       : "cl100k_base",
        "o4-mini"                  : "cl100k_base",
    }

    @classmethod
    def count_tokens(cls, text: str, model: str = None) -> int:
        """テキストのトークン数をカウント"""
        if model is None:
            model = config.get("models.default", "gpt-4o-mini")

        try:
            # モデル別のエンコーディングを取得
            encoding_name = cls.MODEL_ENCODINGS.get(model, "cl100k_base")
            enc = tiktoken.get_encoding(encoding_name)
            return len(enc.encode(text))
        except Exception as e:
            logger.error(f"トークンカウントエラー: {e}")
            # 簡易的な推定（1文字 = 0.5トークン）
            return len(text) // 2

    @classmethod
    def truncate_text(cls, text: str, max_tokens: int, model: str = None) -> str:
        """テキストを指定トークン数に切り詰め"""
        if model is None:
            model = config.get("models.default", "gpt-4o-mini")

        try:
            encoding_name = cls.MODEL_ENCODINGS.get(model, "cl100k_base")
            enc = tiktoken.get_encoding(encoding_name)
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return enc.decode(tokens[:max_tokens])
        except Exception as e:
            logger.error(f"テキスト切り詰めエラー: {e}")
            # 簡易的な切り詰め
            estimated_chars = max_tokens * 2
            return text[:estimated_chars]

    @classmethod
    def estimate_cost(cls, input_tokens: int, output_tokens: int, model: str = None) -> float:
        """API使用コストの推定"""
        if model is None:
            model = config.get("models.default", "gpt-4o-mini")

        # 設定ファイルから料金を取得
        pricing = config.get("model_pricing", {})
        model_pricing = pricing.get(model, pricing.get("gpt-4o-mini"))

        if not model_pricing:
            # デフォルト料金
            model_pricing = {"input": 0.00015, "output": 0.0006}

        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]

        return input_cost + output_cost

    @classmethod
    def get_model_limits(cls, model: str) -> Dict[str, int]:
        """モデルのトークン制限を取得"""
        limits = {
            "gpt-4o"      : {"max_tokens": 128000, "max_output": 4096},
            "gpt-4o-mini" : {"max_tokens": 128000, "max_output": 4096},
            "gpt-4.1"     : {"max_tokens": 128000, "max_output": 4096},
            "gpt-4.1-mini": {"max_tokens": 128000, "max_output": 4096},
            "o1"          : {"max_tokens": 128000, "max_output": 32768},
            "o1-mini"     : {"max_tokens": 128000, "max_output": 65536},
            "o3"          : {"max_tokens": 200000, "max_output": 100000},
            "o3-mini"     : {"max_tokens": 200000, "max_output": 100000},
            "o4"          : {"max_tokens": 256000, "max_output": 128000},
            "o4-mini"     : {"max_tokens": 256000, "max_output": 128000},
        }
        return limits.get(model, {"max_tokens": 128000, "max_output": 4096})


# ==================================================
# レスポンス処理
# ==================================================
class ResponseProcessor:
    """API レスポンスの処理"""

    @staticmethod
    def extract_text(response: Response) -> List[str]:
        """レスポンスからテキストを抽出"""
        texts = []

        if hasattr(response, 'output'):
            for item in response.output:
                if hasattr(item, 'type') and item.type == "message":
                    if hasattr(item, 'content'):
                        for content in item.content:
                            if hasattr(content, 'type') and content.type == "output_text":
                                if hasattr(content, 'text'):
                                    texts.append(content.text)

        # フォールバック: output_text属性
        if not texts and hasattr(response, 'output_text'):
            texts.append(response.output_text)

        return texts

    @staticmethod
    def format_response(response: Response) -> Dict[str, Any]:
        """レスポンスを整形"""
        return {
            "id"        : getattr(response, "id", None),
            "model"     : getattr(response, "model", None),
            "created_at": getattr(response, "created_at", None),
            "text"      : ResponseProcessor.extract_text(response),
            "usage"     : getattr(response, "usage", {}),
        }

    @staticmethod
    def save_response(response: Response, filename: str = None) -> str:
        """レスポンスの保存"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"response_{timestamp}.json"

        formatted = ResponseProcessor.format_response(response)

        # ファイルパスの生成
        logs_dir = Path(config.get("paths.logs_dir", "logs"))
        logs_dir.mkdir(exist_ok=True)
        filepath = logs_dir / filename

        # 保存
        save_json_file(formatted, str(filepath))

        return str(filepath)


# ==================================================
# APIクライアント
# ==================================================
class OpenAIClient:
    """OpenAI API クライアント"""

    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key)

    @error_handler
    @timer
    def create_response(self, messages: List[EasyInputMessageParam], model: str = None, **kwargs) -> Response:
        """Responses API呼び出し"""
        if model is None:
            model = config.get("models.default", "gpt-4o-mini")

        params = {
            "model": model,
            "input": messages,
        }
        params.update(kwargs)

        return self.client.responses.create(**params)

    @error_handler
    @timer
    def create_chat_completion(self, messages: List[ChatCompletionMessageParam], model: str = None, **kwargs):
        """Chat Completions API呼び出し"""
        if model is None:
            model = config.get("models.default", "gpt-4o-mini")

        params = {
            "model"   : model,
            "messages": messages,
        }
        params.update(kwargs)

        return self.client.chat.completions.create(**params)


# ==================================================
# ユーティリティ関数
# ==================================================
def sanitize_key(name: str) -> str:
    """キー用に安全な文字列へ変換"""
    return re.sub(r'[^0-9a-zA-Z_]', '_', name).lower()


def load_json_file(filepath: str) -> Optional[Dict[str, Any]]:
    """JSONファイルの読み込み"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"JSONファイル読み込みエラー: {e}")
        return None


def save_json_file(data: Dict[str, Any], filepath: str) -> bool:
    """JSONファイルの保存"""
    try:
        # ディレクトリの作成
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"JSONファイル保存エラー: {e}")
        return False


def format_timestamp(timestamp: Union[int, float, str] = None) -> str:
    """タイムスタンプのフォーマット"""
    if timestamp is None:
        timestamp = time.time()

    if isinstance(timestamp, str):
        return timestamp

    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def create_session_id() -> str:
    """セッションIDの生成"""
    return hashlib.md5(f"{time.time()}_{id(object())}".encode()).hexdigest()[:8]


# ==================================================
# エクスポート
# ==================================================
__all__ = [
    # 型定義
    'RoleType',

    # クラス
    'ConfigManager',
    'MessageManager',
    'TokenManager',
    'ResponseProcessor',
    'OpenAIClient',

    # デコレータ
    'error_handler',
    'timer',
    'cache_result',

    # ユーティリティ
    'sanitize_key',
    'load_json_file',
    'save_json_file',
    'format_timestamp',
    'create_session_id',
    'clear_cache',

    # 定数
    'developer_text',
    'user_text',
    'assistant_text',

    # グローバル
    'config',
]