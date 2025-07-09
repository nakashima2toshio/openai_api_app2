#
import pandas as pd
from pathlib import Path

def get_cities_list():
    # 1. CSVファイルのパスを指定
    csv_path = Path("data/cities_list.csv")

    # 2. CSVファイルを読み込む（先頭の空白を無視）
    df = pd.read_csv(csv_path, skipinitialspace=True)

    # 3. 列名の空白を削除し、不要な列を削除
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[col for col in df.columns if col.lower().startswith("unnamed")])

    # 4. 'country' 列の値を小文字化し、空白を削除
    df['country'] = df['country'].str.strip().str.lower()

    # 5. 'Japan' または 'japan' に一致する行を抽出
    target_countries = ['Japan', 'japan']
    df_filtered = df[df['country'].isin(target_countries)]

    # 必要に応じてCSVファイルとして保存
    # df_filtered.to_csv("filtered_cities.csv", index=False)
    return df_filtered


def main():
    df_filtered = get_cities_list()
    print(df_filtered)

if __name__ == "__main__":
    main()


