import torch
from satm import SATM
from bow_utils import *
from prepare_sporc import load_sporc

episodes = load_sporc("sporc_turns_selected_clean.csv")

# load pretrained Qwen summaries as dict: episode → summary_text
# 假设你已有 qwen_summaries.json
import json
qsum = json.load(open("qwen_summaries.json"))

# build vocab
all_texts = []
for ep, utts in episodes.items():
    for u in utts:
        all_texts.append(u["text"])
    if ep in qsum:
        all_texts.append(qsum[ep])

vocab = build_vocab(all_texts)
V = len(vocab)

model = SATM(vocab_size=V, Ks=25, Ko=25)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

for step, (ep, utts) in enumerate(episodes.items()):
    if ep not in qsum:
        continue

    bow_dialogue = build_dialogue_bow(utts, vocab)
    bow_summary  = build_summary_bow(qsum[ep], vocab)

    loss, _, _ = model(bow_dialogue, bow_summary)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"step {step}, loss = {loss.item():.2f}")

