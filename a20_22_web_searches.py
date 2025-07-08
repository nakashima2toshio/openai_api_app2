# streamlit run a20_22_web_searches.py  --server.port=8503
# モデルが応答を生成する前に、Webで最新情報を検索できるようにします。
# -----------------------------------------------------
# [WebSearch]
# ・モデル生成文中にネット情報の URL 引用が欲しい
# -----------------------------------------------------
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from openai.types.responses.web_search_tool_param import UserLocation
from openai import OpenAI
from openai.types.responses import(
    WebSearchToolParam,
)
from a0_common_helper.helper import init_page, select_model

import streamlit as st
# --- インポート直後に１度だけ実行する ---
st.set_page_config(
    page_title="ChatGPT Responses API",
    page_icon="2025-5 Nakashima"
)

# --------------------------------------------
#
# --------------------------------------------
def web_search_prev(demo_name):
    client = OpenAI()
    model = select_model()

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    with st.form(key="web_search_form"):
        st.session_state.user_input = st.text_area(
            "東京の最新・天気予報：地域名称の入力:（新宿とか）",
            height=75,
            value=st.session_state.user_input,
            key="user_input_form"
        )
        submit_button = st.form_submit_button(label="送信")

    if submit_button and st.session_state.user_input:
        #  ToolParam型を明示的に使う
        tool = WebSearchToolParam(
            type="web_search_preview",
            user_location=UserLocation(
                type="approximate",
                country="JP",
                city=st.session_state.user_input,
                region="Tokyo"
            )
        )

        response = client.responses.create(
            model=model,
            tools=[tool],
            input="明日の天気は?",
        )
        st.markdown("## 回答")
        st.write(response.output_text)

def sample2():
    pass

# ==================================================
# メインルーティン
# ==================================================
def main() -> None:
    init_page("web searches")
    page_funcs = {
        "web_search_prev": web_search_prev,
        "sample2": sample2,
    }
    demo_name = st.sidebar.radio("デモを選択", list(page_funcs.keys()))
    st.session_state.current_demo = demo_name
    page_funcs[demo_name](demo_name)

if __name__ == "__main__":
    main()

# streamlit run a20_22_web_searches.py  --server.port=8503
# pprint.pprint(response.model_dump())
