import gzip
import json
import pandas as pd

file_path = "speakerTurnData.jsonl.gz"

# 读取前 5 行示例
rows = []
with gzip.open(file_path, "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        rows.append(json.loads(line))

# 转成 DataFrame 方便查看
df = pd.DataFrame(rows)
print("数据集字段（columns）:")
print(df.columns.tolist())

print("\n前几行数据:")
print(df.head(2).to_string())

