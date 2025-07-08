##### a20_00_helper_skeleton.md

helper_api.py (コア機能)


| 機能               | 内容                                 | 備考 |
| ------------------ | ------------------------------------ | ---- |
| 設定管理           | ConfigManager                        |      |
| メッセージ管理     | MessageManager (基本版)              |      |
| トークン管理       | MTokenManager                        |      |
| レスポンス処理     | ResponseProcessor                    |      |
| API クライアント   | OpenAIClient                         |      |
| 汎用ユーティリティ | ファイル操作、キャッシュ、デコレータ |      |

使用方法
```python
# python# Streamlitアプリの場合
from helper_st import *

# API専用スクリプトの場合
from helper_api import MessageManager, TokenManager, OpenAIClient
```

今後の拡張

- helper_audio.py: 音声処理専用機能
- helper_vision.py: 画像処理専用機能
- helper_embeddings.py: 埋め込み処理専用機能
