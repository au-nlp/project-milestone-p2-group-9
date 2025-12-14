# inference_tds.py
import torch
from dataset_tds import SPoRCDataset
from model_tds import TDSModel

def infer_one_episode(ep, ds, model, satm_model, device='cuda', extract_steps=6):
    input_ids, attn_mask, roles, texts = ds.encode_episode(ep)
    input_ids = input_ids.to(device); attn_mask = attn_mask.to(device)
    bow_dialogue = None  # build bow for episode and pass into SATM
    topic_vec = satm_model.get_topic_vector(bow_dialogue).to(device)

    # encode utts
    utt_vecs = model.utt_encoder.encode_utts(input_ids, attn_mask)
    role_embeddings = model.role_emb(roles).to(device)
    chosen, probs = model.extractor(utt_vecs, topic_vec, role_embeddings, gold_indices=None, extract_steps=extract_steps)

    selected_texts = [texts[i] for i in chosen]
    gen = model.generate(selected_texts, max_length=150)
    return {
        "chosen_indices": chosen,
        "selected_texts": selected_texts,
        "generated_summary": gen
    }

if __name__ == "__main__":
    ds = SPoRCDataset("sporc_turns_selected_clean.csv")
    model = TDSModel(bart_name='facebook/bart-large').to('cuda')
    model.load_state_dict(torch.load("tds_ckpt.pt"))
    # satm_model must be loaded
    class DummySATM:
        def get_topic_vector(self, bow):
            return torch.randn(25)
    satm = DummySATM()
    ep = ds.all_episode_ids()[0]
    res = infer_one_episode(ep, ds, model, satm, device='cuda')
    print(res["generated_summary"])

