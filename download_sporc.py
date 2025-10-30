from huggingface_hub import hf_hub_download
import os, shutil

# === 1️⃣ 设置保存目录 ===
save_dir = os.path.expanduser("~/Desktop/semester 3/NLP/project")
os.makedirs(save_dir, exist_ok=True)

# === 2️⃣ 要下载的完整文件名 ===
files = [
    "episodeLevelData.jsonl.gz",   # 约 12 GB
    "speakerTurnData.jsonl.gz"     # 约 10 GB
]

# === 3️⃣ 下载并复制到项目目录 ===
for f in files:
    print(f"⬇️  Downloading {f} ...")
    path = hf_hub_download(repo_id="blitt/SPoRC", filename=f, repo_type="dataset")
    dest = os.path.join(save_dir, f)
    shutil.copy(path, dest)
    print(f"✅ Saved to {dest}")

print("\n🎉 All full data files downloaded successfully!")
