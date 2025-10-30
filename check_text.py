import pandas as pd

# 1️⃣ 读取你刚生成的干净文件
df = pd.read_csv("/Users/allin1307/Desktop/semester 3/NLP/project/sporc_turns_selected_clean.csv")

# 2️⃣ 检查 text 列中空值（None/NaN/空字符串）
empty_mask = df["text"].isna() | (df["text"].str.strip() == "")
num_empty = empty_mask.sum()

# 3️⃣ 检查过短文本（少于 10 个字符）
short_mask = df["text"].str.len() < 10
num_short = short_mask.sum()

# 4️⃣ 打印结果
print("✅ 总样本数:", len(df))
print("🚫 空发言数:", num_empty)
print("⚠️ 太短文本数 (<10 字符):", num_short)

# （可选）查看几个示例
print("\n🧩 示例空发言:")
print(df[empty_mask].head(5))
print("\n🧩 示例短文本:")
print(df[short_mask].head(5))

