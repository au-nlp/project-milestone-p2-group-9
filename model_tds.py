# model_tds.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartTokenizer, BartModel
import math

class RoleEmbedding(nn.Module):
    def __init__(self, role_list=['host','guest','other'], hidden=768):
        super().__init__()
        self.roles = role_list
        self.emb = nn.Embedding(len(role_list), hidden)
        self.role2idx = {r:i for i,r in enumerate(role_list)}

    def forward(self, role_names):
        idxs = [self.role2idx.get(r, self.role2idx['other']) for r in role_names]
        idxs = torch.tensor(idxs, device=self.emb.weight.device)
        return self.emb(idxs)  # [num_utt, hidden]

class UtteranceEncoder(nn.Module):
    def __init__(self, bart_encoder_name='facebook/bart-large', device='cuda'):
        super().__init__()
        self.bart_encoder = BartModel.from_pretrained(bart_encoder_name).get_encoder()
        self.tokenizer = BartTokenizer.from_pretrained(bart_encoder_name)
        self.hidden = self.bart_encoder.config.hidden_size
        self.device=device

    def encode_utts(self, input_ids, attention_mask):
        # input_ids: [num_utt, L]
        bsz = input_ids.size(0)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.bart_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.last_hidden_state: [num_utt, L, hidden]
        # compute sentence vector by mean pooling of token vectors masked
        mask = attention_mask.unsqueeze(-1)
        s = (outputs.last_hidden_state * mask).sum(1) / (mask.sum(1).clamp(min=1e-9))
        return s  # [num_utt, hidden]

class ExtractorPointer(nn.Module):
    def __init__(self, hidden=1024, topic_dim=128):
        super().__init__()
        self.hidden = hidden
        self.topic_proj = nn.Linear(topic_dim, hidden)
        self.query_proj = nn.Linear(hidden, hidden)
        self.key_proj = nn.Linear(hidden, hidden)
        self.gru = nn.GRUCell(hidden, hidden)

    def forward(self, utt_vecs, topic_vec, role_emb, gold_indices=None, extract_steps=5):
        """
        utt_vecs: [N, H]
        topic_vec: [topic_dim]
        role_emb: [N, H]
        gold_indices: list of indices for supervision or None
        """
        device = utt_vecs.device
        topic = self.topic_proj(topic_vec.to(device)).view(-1)  # [H]
        state = nn.Parameter(torch.randn(self.hidden, device=device))  # learned init per forward (could be module param)
        # we can set a fixed init state
        state = state + topic

        if role_emb is not None:
            utts = utt_vecs + role_emb
        else:
            utts = utt_vecs

        probs_seq = []
        loss = 0.0
        chosen = []
        N = utts.size(0)
        for t in range(min(extract_steps, N)):
            q = self.query_proj(state)  # [H]
            k = self.key_proj(utts)      # [N,H]
            scores = torch.matmul(k, q) / math.sqrt(self.hidden)  # [N]
            probs = F.softmax(scores, dim=0)
            probs_seq.append(probs)
            sel = torch.argmax(probs).item()
            chosen.append(sel)

            if gold_indices is not None:
                # supervised: assume gold_indices is list length >= extract_steps
                target = gold_indices[t] if t < len(gold_indices) else gold_indices[-1]
                loss = loss - torch.log(probs[target] + 1e-12)

            # update state
            state = self.gru(utts[sel].unsqueeze(0), state.unsqueeze(0)).squeeze(0)

        if gold_indices is None:
            return chosen, probs_seq
        else:
            return chosen, loss / min(extract_steps, N)

class TopicFusionForRefiner(nn.Module):
    def __init__(self, topic_dim, hidden):
        super().__init__()
        self.fc = nn.Linear(topic_dim, hidden)

    def forward(self, encoder_outputs, topic_vec):
        # encoder_outputs: [N, L, H] or [sum_len, H] -> we propose to add topic to encoder hidden states
        topic_h = self.fc(topic_vec).unsqueeze(0).unsqueeze(0)  # [1,1,H]
        return encoder_outputs + topic_h  # broadcasting

class TDSModel(nn.Module):
    def __init__(self, bart_name='facebook/bart-large', topic_dim=25, role_list=['host','guest','other'], device='cuda'):
        super().__init__()
        self.device = device
        self.utt_encoder = UtteranceEncoder(bart_encoder_name=bart_name, device=device)
        self.role_emb = RoleEmbedding(role_list=role_list, hidden=self.utt_encoder.hidden).to(device)
        self.extractor = ExtractorPointer(hidden=self.utt_encoder.hidden, topic_dim=topic_dim).to(device)

        # Refiner: BART seq2seq
        self.bart = BartForConditionalGeneration.from_pretrained(bart_name).to(device)
        self.tokenizer = BartTokenizer.from_pretrained(bart_name)
        self.topic_fuser = TopicFusionForRefiner(topic_dim, self.utt_encoder.hidden).to(device)

    def forward_train(self, input_ids, attention_mask, roles, topic_vec, gold_extract_idx, gold_summary_text, device='cuda'):
        """
        1) encode utterances -> utt_vecs
        2) extractor supervised -> loss_extract
        3) build refiner input (concatenate selected utters as text)
        4) run BART generation training -> loss_refiner (CrossEntropy)
        """
        # encode utterances
        utt_vecs = self.utt_encoder.encode_utts(input_ids, attention_mask)  # [N, H]
        role_embeddings = self.role_emb(roles).to(device)  # [N,H]
        chosen, loss_ex = self.extractor(utt_vecs, topic_vec, role_embeddings, gold_indices=gold_extract_idx, extract_steps=len(gold_extract_idx))

        # build refiner input text from chosen indices
        # NOTE: the tokenizer is on text side, but this function assumes you passed texts elsewhere
        # For simplicity, we accept gold_summary_text and will create inputs for teacher forcing
        selected_texts = []
        for i in chosen:
            # we'll need raw utter text mapping outside; here keep placeholder
            selected_texts.append("<utt_{}>".format(i))
        input_for_bart = " ".join(selected_texts)

        # Tokenize target summary
        tgt = self.tokenizer(gold_summary_text, return_tensors='pt', truncation=True, max_length=200).to(device)
        src = self.tokenizer(input_for_bart, return_tensors='pt', truncation=True, max_length=512).to(device)

        # fuse topic into encoder hidden states (here we rely on bart encoder)
        outputs = self.bart(input_ids=src['input_ids'], attention_mask=src['attention_mask'], labels=tgt['input_ids'])
        loss_ref = outputs.loss

        return loss_ref, loss_ex, chosen

    def generate(self, selected_texts, max_length=120):
        src = " ".join(selected_texts)
        inputs = self.tokenizer(src, return_tensors='pt', truncation=True, max_length=1024).to(self.device)
        out = self.bart.generate(**inputs, max_length=max_length, num_beams=4)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

