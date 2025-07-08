# utils/get_data_df.py
import pandas as pd
import json

def get_data_df(json_file_name):
    json_file = json_file_name
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(
        list(data.items()),
        columns=['key', 'value']
    )

    return data, df
