# dataset_tds.py
import json
import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoTokenizer


class SPoRCDataset:
    """
    Dataset layer for SPoRC.

    Responsibilities:
    1. Load turn-level CSV and group by episode
    2. Load episode-level summaries from JSONL
    3. Provide role-aware pseudo-labels for extractor training
    4. Enforce training rules:
       - global_summary MUST exist, otherwise skip episode
       - host / guest trained only if corresponding summary exists
    """

    def __init__(
        self,
        csv_path: str,
        summary_jsonl_path: str,
        bart_name: str = "facebook/bart-large",
        sent_emb_name: str = "all-mpnet-base-v2",
        device: str = "cuda",
    ):
        # ---- load CSV ----
        self.df = pd.read_csv(csv_path)
        self.episodes = self._group_by_episode(self.df)

        # ---- tokenizer (used later by models) ----
        self.tokenizer = AutoTokenizer.from_pretrained(bart_name)

        # ---- sentence embedder (for pseudo-labels) ----
        self.sent_embedder = SentenceTransformer(sent_emb_name, device=device)

        # ---- load summaries ----
        self.summaries = {}
        self._load_summaries_jsonl(summary_jsonl_path)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _group_by_episode(self, df: pd.DataFrame):
        episodes = defaultdict(list)

        for _, row in df.iterrows():
            ep = row["episode"]
            episodes[ep].append(
                {
                    "turn": int(row["turn"]),
                    "role": row["role"],
                    "speaker": row["speaker"],
                    "text": row["text"],
                }
            )

        for ep in episodes:
            episodes[ep] = sorted(episodes[ep], key=lambda x: x["turn"])

        return episodes

    def _load_summaries_jsonl(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                ep = obj["episode"]

                self.summaries[ep] = {
                    "global": obj.get("global_summary", "").strip(),
                    "host": obj.get("host_summary", "").strip(),
                    "guest": obj.get("guest_summary", "").strip(),
                }

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def episode_ids(self):
        return list(self.episodes.keys())

    def encode_episode(self, episode: str, max_utt_len: int = 64):
        utts = self.episodes[episode]
        texts = [u["text"] for u in utts]
        roles = [u["role"] for u in utts]

        enc = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_utt_len,
            return_tensors="pt",
        )

        return enc["input_ids"], enc["attention_mask"], roles, texts

    def build_pseudo_labels(
        self,
        episode: str,
        summary_type: str = "global",
        topk: int = 5,
    ):
        """
        Build pseudo-labels for ONE episode and ONE summary type.

        Rules:
        - global_summary MUST exist, otherwise skip episode entirely
        - host / guest trained only if corresponding summary exists
        """

        # ---- episode must have summaries ----
        if episode not in self.summaries:
            print(f"[WARN] episode {episode}: summary not found, skip training")
            return None

        summaries = self.summaries[episode]

        # ---- global summary is mandatory ----
        if summaries["global"] == "":
            print(f"[WARN] episode {episode}: global_summary missing, skip training")
            return None

        # ---- role-specific summary ----
        summary_text = summaries.get(summary_type, "").strip()
        if summary_text == "":
            return None

        # ---- select utterances ----
        utts = self.episodes[episode]

        if summary_type == "host":
            idx_text = [
                (i, u["text"]) for i, u in enumerate(utts) if u["role"] == "host"
            ]
        elif summary_type == "guest":
            idx_text = [
                (i, u["text"]) for i, u in enumerate(utts) if u["role"] == "guest"
            ]
        else:
            idx_text = [(i, u["text"]) for i, u in enumerate(utts)]

        if len(idx_text) == 0:
            return None

        indices, texts = zip(*idx_text)

        # ---- sentence similarity ----
        utt_emb = self.sent_embedder.encode(list(texts), convert_to_tensor=True)
        sum_emb = self.sent_embedder.encode(summary_text, convert_to_tensor=True)

        sims = util.cos_sim(utt_emb, sum_emb).squeeze(1)

        k = min(topk, sims.size(0))
        topk_local = torch.topk(sims, k=k).indices.tolist()

        topk_global = [indices[i] for i in topk_local]
        return topk_global
