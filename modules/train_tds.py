# train_tds.py
import torch
from dataset_tds import SPoRCDataset
from model_tds import TDSModel
import argparse
from tqdm import tqdm
import random
import math

def info_nce_contrastive(utt_vecs, roles, temperature=0.07):
    """
    utt_vecs: [N,H]
    roles: list length N (strings)
    We compute role-level pooled vectors and do InfoNCE such that same-role pairs are closer.
    """
    device = utt_vecs.device
    # pool by role
    role2vecs = {}
    for i,r in enumerate(roles):
        role2vecs.setdefault(r, []).append(utt_vecs[i:i+1])
    role_means = []
    role_keys = []
    for k,v in role2vecs.items():
        stack = torch.cat(v, dim=0)
        role_means.append(stack.mean(0, keepdim=True))
        role_keys.append(k)
    if len(role_means) < 2:
        return torch.tensor(0.0, device=device)

    role_means = torch.cat(role_means, dim=0)  # [R,H]
    role_means = F.normalize(role_means, p=2, dim=-1)
    sim = torch.matmul(role_means, role_means.T) / temperature
    # positive on diagonal, negatives off-diagonal -> use cross entropy
    labels = torch.arange(role_means.size(0), device=device)
    loss = 0.0
    # compute per-row softmax
    loss = - sim.diag().softmax(dim=0).log().sum()  # rough; more standard: treat each as query among others
    # we provide a simplified variant; you can replace with InfoNCE standard implementation
    return loss

def train_loop(csv_path, qwen_json, satm_model, output_ckpt="tds_ckpt.pt", device='cuda'):
    ds = SPoRCDataset(csv_path, qwen_summary_json=qwen_json, device=device)
    episod_ids = ds.all_episode_ids()
    labels = ds.build_pseudo_labels(topk=5)  # pseudo labels
    model = TDSModel(bart_name='facebook/bart-large', topic_dim=25, device=device)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(5):
        random.shuffle(episod_ids)
        pbar = tqdm(episod_ids)
        total_loss = 0.0
        for ep in pbar:
            try:
                input_ids, attn_mask, roles, texts = ds.encode_episode(ep)
            except Exception as e:
                continue
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            # get topic vector from your pretrained SATM (we assume satm_model has method get_topic_vector that returns torch.tensor)
            # build BOW for this episode for satm externally; here we assume satm_model is callable
            bow_dialogue = None  # you need to produce bow vector externally and pass satm_model.get_topic_vector(bow)
            topic_vec = satm_model.get_topic_vector(bow_dialogue).to(device)  # [Ks]

            gold_idx = labels.get(ep, None)
            if gold_idx is None:
                # skip episodes without qwen pseudo-label for now
                continue

            gold_summary = ds.qwen[ep]

            model.train()
            optim.zero_grad()
            loss_ref, loss_ex, chosen = model.forward_train(input_ids, attn_mask, roles, topic_vec, gold_extract_idx=gold_idx, gold_summary_text=gold_summary, device=device)
            # contrastive loss
            utt_vecs = model.utt_encoder.encode_utts(input_ids, attn_mask)
            loss_con = info_nce_contrastive(utt_vecs, roles)

            loss = loss_ref + 1.0 * loss_ex + 0.5 * loss_con
            loss.backward()
            optim.step()

            total_loss += loss.item()
            pbar.set_description(f"ep {ep} loss {loss.item():.4f}")

        print("Epoch", epoch, "avg loss", total_loss / max(1, len(episod_ids)))
        torch.save(model.state_dict(), output_ckpt)

if __name__ == "__main__":
    # satm_model must be implemented and loaded; for demo effect we create a stub
    class DummySATM:
        def get_topic_vector(self, bow):
            return torch.randn(25)
    satm = DummySATM()
    train_loop("sporc_turns_selected_clean.csv", "qwen_summaries.json", satm, output_ckpt="tds_ckpt.pt")

