# python get_url_to_text.py
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List, Tuple, Dict

BASE_URL = "https://openai.github.io/openai-agents-python/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ---------- 1. サイドバーから (title, href) を抽出 ---------- #
def get_sidebar_menu(url: str) -> List[Tuple[str, str]]:
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    sidebar = soup.find("nav", class_="md-nav") or soup.find("nav")
    if sidebar is None:
        raise RuntimeError("サイドバーの <nav> 要素が見つかりませんでした。")

    anchors = sidebar.select("a.md-nav__link[href]") or sidebar.select("a[href]")

    menu: List[Tuple[str, str]] = []
    for a in anchors:
        text = a.get_text(strip=True)
        href = a["href"].strip()
        if not text or not href or href.startswith("#"):
            continue
        menu.append((text, href))

    # 重複排除（順序保持）
    seen, uniq_menu = set(), []
    for item in menu:
        if item not in seen:
            uniq_menu.append(item)
            seen.add(item)
    return uniq_menu


# ---------- 2. 各ページの本文を取得して dict 化 ---------- #
def fetch_all_pages(menu: List[Tuple[str, str]],
                    base_url: str = BASE_URL) -> Dict[str, str]:
    url_text_dict: Dict[str, str] = {}

    for title, href in menu:
        full_url = urljoin(base_url, href)
        try:
            resp = requests.get(full_url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            print(f"⚠️  {title}: {e}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        for t in soup(["script", "style", "noscript"]):
            t.decompose()

        content_div = soup.find("div", class_="md-content")
        target = (content_div.find("article", class_="md-content__inner")
                  if content_div else None) or content_div or soup

        page_text = target.get_text(separator="\n", strip=True)
        url_text_dict[title] = page_text
        print(f"✔ {title}")

    return url_text_dict


# ---------- 実行ブロック ---------- #
if __name__ == "__main__":
    # 1) サイドバー取得
    menu = get_sidebar_menu(BASE_URL)
    print(f"Found {len(menu)} sidebar links.\n")

    # 2) 各ページをクロール
    url_text_dict = fetch_all_pages(menu, BASE_URL)

    # 3) 動作確認：先頭 3 件の冒頭 200 文字を表示
    print("\n===== Preview =====")
    for i, (title, text) in enumerate(url_text_dict.items()):
        print(f"\n--- {title} ---")
        print(text[:200].replace("\n", " "))
        if i == 2:
            break

    # 4) JSON 文字列化して保存（型チェック警告を回避）
    json_str = json.dumps(url_text_dict, ensure_ascii=False, indent=2)
    with open("agents_docs.json", "w", encoding="utf-8") as fp:
        fp.write(json_str)

    print("\nSaved to agents_docs.json ✅")

