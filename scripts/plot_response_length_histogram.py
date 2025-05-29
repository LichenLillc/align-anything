import json
import matplotlib.pyplot as plt

# 加载 JSON 文件
with open("train_30k.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 存储 better 和 worse responses 的长度
better_lengths = []
worse_lengths = []

# 遍历数据并根据 overall_response 判断哪个是 better
for entry in data:
    overall = entry.get("overall_response")
    if int(overall) == 1:
        better = entry["response_1"]
        worse = entry["response_2"]
    elif int(overall) == 2:
        better = entry["response_2"]
        worse = entry["response_1"]
    else:
        continue  # 如果字段异常，跳过

    better_lengths.append(len(better))
    worse_lengths.append(len(worse))

# 绘图
plt.figure(figsize=(10, 6))
plt.hist(better_lengths, bins=30, alpha=0.6, color="blue", label="Better Response Lengths")
plt.hist(worse_lengths, bins=30, alpha=0.6, color="orange", label="Worse Response Lengths")

plt.title("Length Distribution of Better vs. Worse Responses - train_30k")
plt.xlabel("Length (Characters)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)

# 保存和显示图像
plt.tight_layout()
plt.savefig("train_30k_better_vs_worse_response_length_histogram.png", dpi=300)
plt.show()
