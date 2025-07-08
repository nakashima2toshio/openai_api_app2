# streamlit run a20_20_function_calling.py
# https://home.openweathermap.org/
# os.getenv(OPENWEATHER_API_KEY)
# /data/2.5/onecall エンドポイントは「当日を含む最大８日分」のデータを返します。
#!/usr/bin/env python3
# a20_20_function_calling.py

import os
import requests
import pprint
from pathlib import Path
from dotenv import load_dotenv

from helper import append_user_message
# ---------------------------------------
# .env ファイルをロード
# env_path = Path(__file__).parent / ".env"
# load_dotenv(dotenv_path=env_path)
# ---------------------------------------

def get_current_weather(city: str, unit: str = "metric") -> dict:
    """
    指定した都市の現在の天気を取得し、coord(lat, lon) を含む dict を返す。
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise RuntimeError("環境変数 OPENWEATHER_API_KEY が設定されていません。")

    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={api_key}&units={unit}"
    )
    # https:// api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_key}
    res = requests.get(url)
    res.raise_for_status()
    data = res.json()

    return {
        "city": city,
        "temperature": data["main"]["temp"],
        "description": data["weather"][0]["description"],
        "coord": data["coord"]
    }

def get_weekly_forecast(lat: float, lon: float, unit: str = "metric") -> list[dict]:
    # 【無料プラン対応版】---------------------------------------------------
    # 緯度・経度を指定して「5日／3時間毎」の予報を取得し、
    # 日付ごとに平均気温と代表的な天気を計算して返す。
    # Returns:
    #   [{"date": "YYYY-MM-DD", "temp_avg": float, "weather": str}, ...]
    # --------------------------------------------------------------------
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise RuntimeError("環境変数 OPENWEATHER_API_KEY が設定されていません。")

    # 無料プランで使える 5day/3hour forecast
    url = (
        f"http://api.openweathermap.org/data/2.5/forecast"
        f"?lat={lat}&lon={lon}"
        f"&units={unit}&appid={api_key}"
    )
    res = requests.get(url)
    res.raise_for_status()
    data = res.json()

    if "list" not in data:
        err = data.get("message", f"HTTP {res.status_code}")
        raise RuntimeError(f"OpenWeather 5日予報エラー: {err}")

    # 日付ごとに分けて平均気温を計算
    daily = {}
    for item in data["list"]:
        date = item["dt_txt"].split(" ")[0]
        temp = item["main"]["temp"]
        weather = item["weather"][0]["description"]
        if date not in daily:
            daily[date] = {"temps": [], "weather": weather}
        daily[date]["temps"].append(temp)

    forecast = []
    for date, info in daily.items():
        avg_temp = sum(info["temps"]) / len(info["temps"])
        forecast.append({
            "date": date,
            "temp_avg": round(avg_temp, 1),
            "weather": info["weather"]
        })

    return forecast

def main():
    city = "kawasaki"
    unit = "metric"

    # 本日の天気
    today = get_current_weather(city, unit)
    print("----- 本日の天気 -----")
    print(f"都市       : {today['city']}")
    print(f"気温       : {today['temperature']}℃")
    print(f"説明       : {today['description']}\n")

    # 5日予報
    coord = today["coord"]
    week = get_weekly_forecast(coord["lat"], coord["lon"], unit)

    print("----- 5日間予報 （3時間毎を日別平均） -----")
    for day in week:
        print(f"{day['date']} : {day['temp_avg']}℃, {day['weather']}")

if __name__ == "__main__":
    main()
