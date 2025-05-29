import pandas as pd
import json
import sys

def parquet_to_json(parquet_path, json_path):
    # 读取 Parquet 文件
    df = pd.read_parquet(parquet_path)
    
    # 将 DataFrame 转换为字典列表
    records = df.to_dict(orient='records')
    
    # 写入 JSON 文件（带缩进和换行）
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python convert.py input.parquet output.json")
        sys.exit(1)
    parquet_to_json(sys.argv[1], sys.argv[2])
