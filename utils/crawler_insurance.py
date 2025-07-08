# crawler_insurance.py
"""
4. Todo（データ化フロー）
フェーズ	作業内容	ヒント
① スキーマ設計	id / category / subcategory / question / answer / source_url / retrieved_at などを決定	添字・タグを入れると後処理が楽
② URL リスト化	上記表＋追加候補をスプレッドシートで管理（重複・死リンク除外）	Chrome 拡張 or Python で自動収集
③ スクレイピング	Python + BeautifulSoup/Selenium で抽出
動的ページは Selenium, HeadlessChrome	日本語テキストは UTF-8 保存
④ クリーニング	・HTMLタグ除去
・改行 → スペース統一
・Q&A ペア抽出
・同義質問の正規化	正規表現＋LangChain/TextLint
⑤ メタ付与	カテゴリ分類、自社 / 第三者視点、更新日などのメタデータを付ける	ルール + 手動レビュー併用
⑥ データ検証	Q≠A ズレや重複をチェック。ランダムサンプリングで人手確認。	pytest + manual spot-check
⑦ フォーマット出力	JSONL / CSV / SQLite など用途別にエクスポート	AI モデル学習なら JSONL が扱いやすい
⑧ 継続運用	cron + GitHub Actions で定期クロール → PR 形式で差分レビュー	「retrieved_at」で更新管理
"""
# ---------------------------------
# crawler_insurance.py（抜粋）
# ---------------------------------
from enum import Enum
from typing import List
from pydantic import BaseModel, HttpUrl, Field


class Category(str, Enum):
    """生命保険 Q&A 用カテゴリ列挙型
    値 = 日本語（データで使うラベル）
    メンバー名 = 英字（Python で扱いやすい識別子）"""
    SELECTION   = "保険選び"
    ELIGIBILITY = "契約可否"
    BENEFICIARY = "受取人"
    COMPLAINT   = "相談・苦情"

    @classmethod
    def values(cls) -> list[str]:
        """['保険選び', '契約可否', …] の形で全値を取得"""
        return [c.value for c in cls]


class InsuranceQA(BaseModel):
    id: int = Field(..., ge=1, description="一意 ID")
    category: Category  # ← str|Category ではなく Enum 固定
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    source: HttpUrl


class InsuranceQADataset(BaseModel):
    items: List[InsuranceQA]

    model_config = {
        "title": "生命保険Q&Aデータセット",
        "description": "InsuranceQA の一覧を保持する"
    }


# ----------------------------
# 動作確認サンプル
# ----------------------------
if __name__ == "__main__":
    raw_data = [
        {
            "id": 1,
            "category": "保険選び",
            "question": "生命保険の選び方にポイントはあるの？",
            "answer": "家族構成とライフプランから①必要保障②期間③金額④保険料負担の４点をチェックする。",
            "source": "https://www.jili.or.jp/knows_learns/q_a/life_insurance/121.html"
        },
        {
            "id"      : 2,
            "category": "契約可否",
            "question": "健康上問題があると生命保険は契約できないの？",
            "answer"  : "症状によっては拒否されるが、完治後一定期間経過や割増保険料・保険金削減条件付きで契約できることがある。緩和型商品も選択肢。",
            "source"  : "https://www.jili.or.jp/knows_learns/q_a/life_insurance/145.html"
        },
        {
            "id"      : 3,
            "category": "受取人",
            "question": "死亡保険金受取人が被保険者より先に死亡していた場合、保険金は誰が受け取る？",
            "answer"  : "一般には受取人の法定相続人が新しい受取人になる。複数なら均等分割。商品により“被保険者の遺族”と定義する場合もある。",
            "source"  : "https://www.jili.or.jp/knows_learns/q_a/life_insurance/9328.html"
        },
        {
            "id"      : 4,
            "category": "相談・苦情",
            "question": "生命保険相談所はどういった機関ですか？",
            "answer"  : "保険業法に基づく指定紛争解決機関で、民間生保に関する一般相談・苦情を無料で受け付ける。会社との話合いが不調なら苦情取次を行う。",
            "source"  : "https://www.seiho.or.jp/contact/about/qanda/"
        },
    ]

    dataset = InsuranceQADataset.model_validate({"items": raw_data})
    print(dataset.model_dump_json(indent=2, ensure_ascii=False))
