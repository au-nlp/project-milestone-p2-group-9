import os, json, gzip, csv
from tqdm import tqdm

BASE = os.path.expanduser("~/Desktop/semester 3/NLP/project")
src_jsonl = os.path.join(BASE, "speakerTurnData.jsonl")
src_gz    = os.path.join(BASE, "speakerTurnData.jsonl.gz")
dst_csv   = os.path.join(BASE, "sporc_turns_clean.csv")

# 1) 选择数据源：优先使用已解压的 .jsonl
if os.path.exists(src_jsonl):
    src = src_jsonl
    is_gz = False
    total_bytes = os.path.getsize(src_jsonl)
else:
    src = src_gz
    is_gz = True
    total_bytes = None  # gz 压缩流无法准确估算总字节

print(f"📚 Source: {src}")
print(f"📄 Output: {dst_csv}")
if os.path.exists(dst_csv):
    raise SystemExit(f"❗输出文件已存在：{dst_csv}，为避免混淆请先删除再运行。")

# 2) 打开输入与输出（逐行处理，避免吃内存）
open_in = (lambda p: open(p, "r", encoding="utf-8", errors="ignore")) if not is_gz \
          else (lambda p: gzip.open(p, "rt", encoding="utf-8", errors="ignore"))

with open_in(src) as fin, open(dst_csv, "w", newline="", encoding="utf-8") as fout:
    writer = csv.writer(fout)
    # 写表头（只保留训练常用字段；后续需要可以再加）
    writer.writerow(["text", "role", "speaker", "episode", "turn"])

    if not is_gz:
        # .jsonl 可用字节进度条（更准确）
        pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Converting (.jsonl)")
        bytes_read = 0
        for line in fin:
            bytes_read += len(line.encode("utf-8", errors="ignore"))
            pbar.update(len(line.encode("utf-8", errors="ignore")))
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            writer.writerow([
                d.get("turnText", ""),
                d.get("inferredSpeakerRole", ""),
                d.get("inferredSpeakerName", ""),
                d.get("mp3url", ""),
                d.get("turnCount", None),
            ])
        pbar.close()
    else:
        # .jsonl.gz 用行进度条（估计 350 万行）
        pbar = tqdm(fin, desc="Converting (.jsonl.gz)", total=3_500_000)
        for line in pbar:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            writer.writerow([
                d.get("turnText", ""),
                d.get("inferredSpeakerRole", ""),
                d.get("inferredSpeakerName", ""),
                d.get("mp3url", ""),
                d.get("turnCount", None),
            ])

