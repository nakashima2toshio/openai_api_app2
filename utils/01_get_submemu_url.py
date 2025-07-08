# streamlit run 01_get_submenu_url.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
from io import StringIO

# Streamlitページ設定
st.set_page_config(page_title="メニューURL抽出", page_icon="🔍")

st.header("URLからメニューとリンクをCSVで取得")

# URL入力
url = st.text_input("対象のURLを入力してください:", value="https://openai.github.io/openai-agents-python/")

# CSV生成関数
def extract_sidebar_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    sidebar = soup.find('nav', class_='menu')
    if not sidebar:
        st.error("メニューが見つかりませんでした。")
        return None

    links = sidebar.find_all('a', class_='menu__link')

    data = []
    for link in links:
        text = link.get_text(strip=True).replace('/', '_').replace(' ', '_')
        full_url = urljoin(url, link.get('href'))
        data.append([text, full_url])

    df = pd.DataFrame(data, columns=['タイトル_サブタイトル', 'URL'])
    return df

# CSV表示とダウンロード処理
if st.button("CSVデータ生成"):
    df = extract_sidebar_links(url)
    if df is not None:
        st.write("取得したデータ:")
        st.dataframe(df)

        # CSVとしてダウンロード
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_str = csv_buffer.getvalue()

        st.download_button(
            label="CSVをダウンロード",
            data=csv_str,
            file_name='menu_links.csv',
            mime='text/csv',
        )
