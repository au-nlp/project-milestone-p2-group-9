import gzip
import json

input_path = "episodeLevelData.jsonl.gz"
output_path = "episodeLevelData.jsonl"

count = 0
with gzip.open(input_path, "rt", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        fout.write(line)
        count += 1
        if count % 10000 == 0:
            print(f"✅ Processed {count:,} examples...")

print(f"\n🎉 Done! Total {count:,} episodes written to {output_path}")
