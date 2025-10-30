import pandas as pd
from tqdm import tqdm
import os

# === 文件路径 ===
base = "/Users/allin1307/Desktop/semester 3/NLP/project"
input_path = os.path.join(base, "sporc_turns_selected.csv")
output_path = os.path.join(base, "sporc_turns_selected_clean.csv")

print(f"📂 输入文件: {input_path}")
print(f"💾 输出文件: {output_path}")

# === 逐块读取，节省内存 ===
chunksize = 500_000   # 每次读取 50 万行，可根据内存调整
filtered_chunks = []
kept, dropped = 0, 0

for chunk in tqdm(pd.read_csv(input_path, chunksize=chunksize), desc="Filtering by role"):
    # 只保留 host 和 guest
    mask = chunk["role"].isin(["host", "guest"])
    filtered_chunk = chunk[mask]
    kept += len(filtered_chunk)
    dropped += len(chunk) - len(filtered_chunk)
    
    # 写入输出文件（分块追加）
    if not os.path.exists(output_path):
        filtered_chunk.to_csv(output_path, index=False, mode="w")
    else:
        filtered_chunk.to_csv(output_path, index=False, mode="a", header=False)

print("\n✅ 过滤完成！")
print(f"保留样本数: {kept:,}")
print(f"剔除样本数: {dropped:,}")
print(f"输出文件路径: {output_path}")

