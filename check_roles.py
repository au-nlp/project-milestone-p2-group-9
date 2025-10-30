import pandas as pd

df = pd.read_csv("/Users/allin1307/Desktop/semester 3/NLP/project/sporc_turns_clean.csv", usecols=["role"])

print("\n角色分布：")
print(df["role"].value_counts())

print("\n所有唯一角色标签：")
print(df["role"].unique())

