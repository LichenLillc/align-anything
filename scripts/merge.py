import json
import pandas as pd

# 加载第一个JSON文件（假设作为better_response）
with open('dpo_1k_5k5ep_generation_results.json', 'r', encoding='utf-8') as f:
    data1 = json.load(f)

# 加载第二个JSON文件（假设作为worse_response）
with open('dpo_1k_base_generation_results.json', 'r', encoding='utf-8') as f:
    data2 = json.load(f)

# 合并数据
merged_data = []
for item1, item2 in zip(data1, data2):
    # 假设每个item都有"instruction"和"response"字段
    merged_item = {
        "question": item1["instruction"],  # 或item2["instruction"]
        "response_1": item1["output"],
        "response_2": item2["output"],
        "overall_response": 1

    }
    merged_data.append(merged_item)

# 保存合并后的数据
with open('5k5ep_base_merged_response.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print("合并完成，结果已保存为merged_data.json")

df = pd.DataFrame(merged_data)
df.to_parquet('/data/align_anything_dpo_5k5ep/val.parquet', engine='pyarrow')

