##### Menu
##### a20_01_responses_parse.py
##### [Messages]
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

# -----------------------------------------------
# "01_00 responses.parseの基本"        : responses_parse_basic,
# -----------------------------------------------
class UserInfo(BaseModel):
    name: str
    age: int
    city: str

# ルートを object にし、その中に配列フィールドを置く
class People(BaseModel):
    users: list[UserInfo]

response = client.responses.parse(
    model=model,
    input=messages,
    text_format=People
)

people: People = response.output_parsed

# -----------------------------------------------
# "01_01  Responsesサンプル(One Shot)" : responses_sample,
# -----------------------------------------------
messages.append(
    EasyInputMessageParam(
        role="user",
        content=user_input
        )
    )

res = client.responses.create(
    model=model,
    input=messages
    )

# -----------------------------------------------
# "01_011 Responsesサンプル(History)"   : responses_memory_sample,
# -----------------------------------------------
res = client.responses.create(
    model=model,
    input=st.session_state.responses_memory_history
    )

for txt in extract_text_from_response(res)

# -----------------------------------------------
# "01_02  画像入力(URL)"               : responses_01_02_passing_url,
# -----------------------------------------------
messages.append(
    EasyInputMessageParam(
        role="user",
        content=[
            ResponseInputTextParam(type="input_text", text=question_text),
            ResponseInputImageParam(type="input_image", image_url=image_url, detail="auto"),
        ],
    )
)

res = client.responses.create(
    model=model,
    input=messages
    )

# -----------------------------------------------
# "01_021 画像入力(base64)"            : responses_01_021_base64_image,
# -----------------------------------------------
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

res = client.responses.create(
    model=model,
    input=messages
    )

# -----------------------------------------------
# "01_03  構造化出力-responses"        : responses_01_03_structured_output,
# -----------------------------------------------
# ------------- Pydantic モデル -------------
class Event(BaseModel):
    name: str
    date: str
    participants: list[str]

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

res = client.responses.create(
    model=model,
    input=messages,
    text=text_cfg
    )

# 5. Pydantic でバリデート
event: Event = Event.model_validate_json(res.output_text)

# -----------------------------------------------
# "01_031 構造化出力-parse"            : responses_01_031_structured_output,
# -----------------------------------------------
# --- 1. Pydantic モデル ---
class Event2(BaseModel):
    name: str
    date: str
    participants: list[str]

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

res = client.responses.parse(
    model=model,
    input=messages,
    text_format=Event2,  # ← ここがポイント
)

# 4. 返却は自動で Event2 に！
event: Event2 = res.output_parsed  # SDK が生成

# -----------------------------------------------
# "01_04  関数 calling"                : responses_01_04_function_calling,
# -----------------------------------------------
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



# -----------------------------------------------
# "01_05  会話状態"                    : responses_01_05_conversation,
# -----------------------------------------------


# -----------------------------------------------
# "01_06  ツール:FileSearch, WebSearch": responses_01_06_tools_file_search,
# -----------------------------------------------
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

# -----------------------------------------------
# "01_061 File Search"                 : responses_01_061_filesearch,
# -----------------------------------------------
vector_store_id = 'XXXXXXXX'

fs_tool = FileSearchToolParam(
    type="file_search",
    vector_store_ids=[vector_store_id],
    max_num_results=20
)

resp = client.responses.create(
    model=model,
    tools=[fs_tool],
    input="請求書の支払い期限は？",
    include=["file_search_call.results"]
)

st.write(resp.output_text)

# -----------------------------------------------
# "01_062 Web Search"                  : responses_01_062_websearch,
# -----------------------------------------------

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

response = client.responses.create(
    model=model,
    tools=[ws_tool],
    input="週末の東京の天気とおすすめの屋内アクティビティは？"
)

st.write(response.output_text)

# -----------------------------------------------
# "01_07  Computer Use Tool Param"     : responses_01_07_computer_use_tool_param,
# -----------------------------------------------

# -----------------------------------------------