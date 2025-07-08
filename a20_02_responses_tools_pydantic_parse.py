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

import os
from typing import List
import requests
import pprint
from openai import OpenAI
from openai.types.responses import EasyInputMessageParam, ResponseInputTextParam

from pydantic import BaseModel
from openai import pydantic_function_tool
# from  openai.lib._tools import pydantic_function_tool

from a0_common_helper.helper import append_user_message
from a1_core_concept.a1_20__moderations import get_default_messages


# ----------------------------------------------------
# 01_01: 基本的な function_call の structured output
# ----------------------------------------------------
class WeatherRequest(BaseModel):
    city: str
    date: str

class NewsRequest(BaseModel):
    topic: str
    date: str

def sample_01_01():
    client = OpenAI()
    response = client.responses.parse(
        model="gpt-4.1",
        input=append_user_message("東京と大阪の明日の天気と、AIの最新ニュースを教えて"),
        tools=[
            pydantic_function_tool(WeatherRequest),
            pydantic_function_tool(NewsRequest)
        ]
    )
    # get_current_weather(city: str, unit: str = "metric")
    city_coords = {
        "東京": {"lat": 35.6895, "lon": 139.69171},
        "大阪": {"lat": 34.6937, "lon": 135.5023}
    }

    # ---------------------
    # 以下、後処理
    # ---------------------
    city_date = dict()
    # 指定した都市の現在の天気を取得し、coord(lat, lon) を含む dict を返す。
    for function_call in response.output:
        print("関数名:", function_call.name)
        print("引数:", function_call.parsed_arguments)
        if hasattr(function_call.parsed_arguments, "city") and hasattr(function_call.parsed_arguments, "date"):
            city = function_call.parsed_arguments.city
            date = function_call.parsed_arguments.date
            print(f"city: {city}, date: {date}")
            city_date[city] = date

            coords = city_coords.get(city)
            if coords:
                lat = coords["lat"]
                lon = coords["lon"]
                API_key = os.getenv("OPENWEATHER_API_KEY")
                url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_key}"
                res = requests.get(url)
                pprint.pprint(res.json())

                unit = "metric"
                url2 = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units={unit}&appid={API_key}"
                res2 = requests.get(url2)
                pprint.pprint(res2.json())
            else:
                print(f"{city}の緯度経度情報がありません。")

# ----------------------------------------------------
# 01_02: 複数ツールの登録・複数関数呼び出し
# ----------------------------------------------------
def sample_01_02():
    client = OpenAI()
    messages = get_default_messages()
    append_user_text = "東京の明日の天気と、AIの最新ニュースを教えて"
    messages.append(
        EasyInputMessageParam(
            role="user",
            content=[
                ResponseInputTextParam(type="input_text", text=append_user_text),
            ]
        )
    )
    response = client.responses.parse(
        model="gpt-4.1",
        input=messages,
        tools=[
            pydantic_function_tool(WeatherRequest),
            pydantic_function_tool(NewsRequest)
        ]
    )

    for function_call in response.output:
        print("関数名:", function_call.name)
        print("引数:", function_call.parsed_arguments)

# ----------------------------------------------------
# 01_021: 複数ツールの登録・複数関数呼び出し
# ----------------------------------------------------
# 1. ツール（Function）のパラメータ定義 ------------------
class CalculatorRequest(BaseModel):
    # 計算式を受け取るツール"""
    exp: str  # 例: "2+2"

class FAQSearchRequest(BaseModel):
    # FAQ 検索クエリを受け取るツール"""
    query: str

# --------------------------------------
# 2. 各ツールのローカル実装（サンプル）
# --------------------------------------
def calculator(exp: str) -> str:
    """与えられた計算式を安全に eval する簡易版"""
    try:
        return str(eval(exp))
    except Exception as e:
        return f"計算エラー: {e}"


def faq_search(query: str) -> str:
    # FAQ DB を検索する代わりにダミー回答を返す"""
    return f"FAQ回答: {query} ...（ここに検索結果が入る）"


def sample_01_021():
    client = OpenAI()
    messages = get_default_messages()
    append_user_text = "2+2 はいくつですか？またはFAQから確認してください。"
    messages.append(
        EasyInputMessageParam(
            role="user",
            content=[
                ResponseInputTextParam(type="input_text", text=append_user_text),
            ]
        )
    )
    response = client.responses.parse(
        model="gpt-4.1",
        input=messages,
        tools=[
            pydantic_function_tool(CalculatorRequest, name="calculator"),
            pydantic_function_tool(FAQSearchRequest, name="faq_search"),
        ],
    )

    # --------------------------------------
    # GPT から返ってきた function_call を実行
    # --------------------------------------
    for function_call in response.output:
        print("関数名:", function_call.name)

        args = function_call.parsed_arguments
        print("引数:", args)

        # 安全な型チェックとモデルへの変換
        if isinstance(args, BaseModel):
            args_dict = args.model_dump()
        else:
            if function_call.name == "calculator":
                args_dict = CalculatorRequest.model_validate(args).model_dump()
            elif function_call.name == "faq_search":
                args_dict = FAQSearchRequest.model_validate(args).model_dump()
            else:
                print("未対応のツールが呼び出されました")
                continue

        if function_call.name == "calculator":
            result = calculator(**args_dict)
            print("計算結果:", result)

        elif function_call.name == "faq_search":
            result = faq_search(**args_dict)
            print("FAQ検索結果:", result)

        else:
            print("未対応のツールが呼び出されました")

# ----------------------------------------------------
# (01_03) ユーザー独自の複雑な構造体（入れ子あり）
# ----------------------------------------------------
def sample_01_03():
    from typing import List
    from pydantic import BaseModel
    from openai import OpenAI

    class Task(BaseModel):
        name: str
        deadline: str

    class ProjectRequest(BaseModel):
        project_name: str
        tasks: List[Task]

    client = OpenAI()
    messages = get_default_messages()
    append_user_text = "プロジェクト『AI開発』には「設計（明日まで）」「実装（来週まで）」というタスクがある"
    messages.append(
        EasyInputMessageParam(
            role="user",
            content=[
                ResponseInputTextParam(type="input_text", text=append_user_text),
            ]
        )
    )
    response = client.responses.parse(
        model="gpt-4.1",
        input=messages,
        tools=[pydantic_function_tool(ProjectRequest)]
    )

    function_call = response.output[0]
    print("関数名:", function_call.name)
    print("引数:", function_call.parsed_arguments)
    # parsed_arguments.tasks は Taskのリストとしてパース

# ----------------------------------------------------
# (01_04) Enum型や型安全なオプションパラメータ付き
# ----------------------------------------------------
def sample_01_04():
    from pydantic import BaseModel
    from enum import Enum
    from openai import OpenAI

    class Unit(str, Enum):
        celsius = "celsius"
        fahrenheit = "fahrenheit"

    class WeatherRequest(BaseModel):
        city: str
        date: str
        unit: Unit

    client = OpenAI()
    messages = get_default_messages()
    append_user_text = "ニューヨークの明日の天気を華氏で教えて"
    messages.append(
        EasyInputMessageParam(
            role="user",
            content=[
                ResponseInputTextParam(type="input_text", text=append_user_text),
            ]
        )
    )
    response = client.responses.parse(
        model="gpt-4.1",
        input=messages,
        tools=[pydantic_function_tool(WeatherRequest)]
    )

    function_call = response.output[0]
    print("関数名:", function_call.name)
    print("引数:", function_call.parsed_arguments)  # unit=Unit.fahrenheit など


# ----------------------------------------------------
# (01_05) text_format引数で自然文のstructured outputを生成
# ----------------------------------------------------
class Step(BaseModel):
    explanation: str
    output: str

class MathResponse(BaseModel):
    steps: List[Step]
    final_answer: str

def sample_01_05():
    client = OpenAI()
    messages = get_default_messages()
    append_user_text = "8x + 31 = 2 を解いてください。途中計算も教えて"
    messages.append(
        EasyInputMessageParam(
            role="developer",
            content=[
                ResponseInputTextParam(type="input_text", text=append_user_text),
            ]
        )
    )
    response = client.responses.parse(
        model="gpt-4.1",
        input=messages,
        text_format=MathResponse,
    )

    for output in response.output:
        if output.type == "message":
            for item in output.content:
                if item.type == "output_text" and item.parsed:
                    print(item.parsed)  # MathResponse構造体
                    print("answer:", item.parsed.final_answer)


# ----------------------------------------------------
# (02_01) 基本パターン（シンプルな構造化データ抽出）
# ----------------------------------------------------
def sample_02_01():
    from pydantic import BaseModel
    from openai import OpenAI

    class PersonInfo(BaseModel):
        name: str
        age: int

    client = OpenAI()
    messages = get_default_messages()
    append_user_text = "彼女の名前は中島美咲で年齢は27歳です。"
    messages.append(
        EasyInputMessageParam(
            role="developer",
            content=[
                ResponseInputTextParam(type="input_text", text=append_user_text),
            ]
        )
    )
    response = client.responses.parse(
        model="gpt-4.1",
        input=messages,
        text_format=PersonInfo,
    )

    person = response.output[0].content[0].parsed
    print(f"名前: {person.name}, 年齢: {person.age}")

# ----------------------------------------------------
# (02_02) 複数エンティティの構造化データ抽出サンプル
# ----------------------------------------------------
# 目的: 1 回のプロンプトから「人物情報」と「書籍情報」の 2 種類を同時に抽出する例。
#       List[PersonInfo] + List[BookInfo] を含む複合 Pydantic モデルで受け取る。
# ------------------------------------
from typing import List
from pydantic import BaseModel
from openai import OpenAI

# ------------------------------------
# 1. 抽出したいデータの Pydantic モデル定義
# ------------------------------------
class PersonInfo(BaseModel):
    name: str
    age: int

class BookInfo(BaseModel):
    title: str
    author: str
    year: int

class ExtractedData(BaseModel):
    persons: List[PersonInfo]
    books: List[BookInfo]

def sample_02_011():
    # ------------------------------------
    # 2. 入力テキスト（複数エンティティを含む）
    # ------------------------------------
    text = """登場人物:
    - 中島美咲 (27歳)
    - 田中亮 (34歳)
    
    おすすめ本:
    1. 『流浪の月』   著者: 凪良ゆう  (2019年)
    2. 『気分上々』   著者: 山田悠介 (2023年)
    """

    # ------------------------------------
    # 3. OpenAI Responses API 呼び出し
    # ------------------------------------
    client = OpenAI()
    response = client.responses.parse(
        model="gpt-4.1",
        input=text,
        text_format=ExtractedData,
    )

    # ------------------------------------
    # 4. 結果の利用例
    # ------------------------------------
    extracted: ExtractedData = response.output[0].content[0].parsed

    print("--- 人物一覧 ---")
    for p in extracted.persons:
        print(f"名前: {p.name}, 年齢: {p.age}")

    print("\n--- 書籍一覧 ---")
    for b in extracted.books:
        print(f"タイトル: {b.title}, 著者: {b.author}, 年: {b.year}")

# ----------------------------------------------------
# (02_02) 複雑なクエリパターン（条件・ソートなど）
# ----------------------------------------------------
def sample_02_02():
    from enum import Enum
    from typing import List, Union
    from pydantic import BaseModel
    from openai import OpenAI

    client = OpenAI()

    class Operator(str, Enum):
        eq = "="
        ne = "!="
        gt = ">"
        lt = "<"

    class Condition(BaseModel):
        column: str
        operator: Operator
        value: Union[str, int]

    class Query(BaseModel):
        table: str
        conditions: List[Condition]
        sort_by: str
        ascending: bool

    messages = get_default_messages()
    append_user_text = "ユーザーテーブルから年齢が20歳以上で東京在住の人を名前で昇順にソートして"
    messages.append(
        EasyInputMessageParam(
            role="developer",
            content=[
                ResponseInputTextParam(type="input_text", text=append_user_text),
            ]
        )
    )
    response = client.responses.parse(
        model="gpt-4.1",
        input=messages,
        text_format=Query,
    )

    query = response.output[0].content[0].parsed
    print(query)


# ----------------------------------------------------
# (02_03) 列挙型・動的な値の利用パターン
# ----------------------------------------------------
def sample_02_03():
    from enum import Enum
    from pydantic import BaseModel
    from openai import OpenAI

    client = OpenAI()

    class Priority(str, Enum):
        high = "高"
        medium = "中"
        low = "低"

    class Task(BaseModel):
        description: str
        priority: Priority

    response = client.responses.parse(
        input="サーバーの再起動を最優先でお願い",
        model="gpt-4.1",
        text_format=Task,
    )

    task = response.output[0].content[0].parsed
    print(task)

# ----------------------------------------------------
# (02_04) Chain of thought: 階層化された出力構造（Nested Structure）
# ----------------------------------------------------
def sample_02_04():
    from typing import List
    from pydantic import BaseModel
    from openai import OpenAI

    class Step(BaseModel):
        explanation: str
        output: str

    class MathSolution(BaseModel):
        steps: List[Step]
        answer: str

    # 推論可能なOpenAIモデル一覧（2025年5月時点）
    # GPT-4.5, GPT-4.1 / GPT-4.1 mini / GPT-4.1 nano, o3 / o3-mini
    client = OpenAI()
    # input_text = """python, streamlit, openaiのAPIの環境で、Question and Answerのアプリを作りたい。
    # コードを作成する手順とそれぞれの手順を提案しなさい。
    # """
    input_text = "美味しいチョコレートケーキを作りたい。"
    response = client.responses.parse(
        input=append_user_message(input_text),
        model="gpt-4.1",  #"gpt-4o",
        text_format=MathSolution,
    )

    solution = response.output[0].content[0].parsed
    todo_no = 0
    for step in solution.steps:
        print(f"\n手順（{todo_no}） ---------------------")
        print(step.explanation, "→", step.output)
        todo_no += 1
    print("最終解:", solution.answer)

# ----------------------------------------------------
# (02_05) 会話履歴を持った連続した構造化出力の処理
# ----------------------------------------------------
def sample_02_05():
    client = OpenAI()

    class QAResponse(BaseModel):
        question: str
        answer: str

    history = []

    def ask_question(question_text):
        response = client.responses.parse(
            input=question_text,
            model="gpt-4.1",
            text_format=QAResponse,
        )
        qa = response.output[0].content[0].parsed
        history.append(qa)
        return qa

    qa1 = ask_question("Pythonの用途を教えてください")
    print(qa1.question, "→", qa1.answer)

    qa2 = ask_question("その中で特にWeb開発に使われるフレームワークは？")
    print(qa2.question, "→", qa2.answer)

    print("会話履歴：")
    for item in history:
        print("-", item.question, "→", item.answer)


def main():
    sample_01_01()
    sample_01_02()
    sample_01_021()
    sample_01_03()
    sample_01_04()
    sample_01_05()

    sample_02_01()
    sample_02_011()
    sample_02_02()
    sample_02_03()
    sample_02_04()
    sample_02_05()

if __name__ == '__main__':
    main()
