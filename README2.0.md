
# Role-Aware Podcast Summarization and Divergence Detection

## Abstract
Podcasts are rich sources of spontaneous, multi-speaker dialogue that mix storytelling, debate, and information sharing. Yet, understanding long podcast episodes remains challenging for listeners and for downstream NLP applications.  
Our project aims to build a **role-aware podcast understanding(QA/RAG) system** that generates both **global** and **role-specific** summaries, while also detecting **divergent conversational segments**, meaning moments when the host and guest disagree or shift topics. Specifically, our system aims to:
1. Retrieve and summarize podcast content relevant to a given topic query (e.g., “AI and education”).

2. Answer open-ended user questions (e.g., “Will AI exacerbate inequality in schools?”) by summarizing different speakers’ perspectives across episodes

3. Detect points of agreement or disagreement between speakers as part of the summarization process.

We use the **SPoRC dataset**, a large-scale corpus of podcast transcripts with speaker-turn annotations, to model host-guest dynamics. Through embedding-based discourse analysis and transformer-based summarization models, we intend to create structured insights that reflect *who said what* and *where the conversation diverged*. This project demonstrates the feasibility of scalable dialogue understanding for complex multi-speaker audio data.

---

## Contributions and Novelty
1. **Research Content**
   
   The project expands from traditional "podcast summarization" to a higher level of dialogue understanding, incorporating character stances and viewpoint disagreements into podcast content analysis, providing a new approach to the structured understanding of multi-character dialogue corpora.

2. **Integration Challenges**
   
   The project integrates multi-level language understanding tasks, including topic identification, character summarization, and viewpoint conflict determination. These three areas originally belonged to different research directions: topic analysis focuses on discourse organization, character summarization on narrative structure, and stance identification involves semantic comparison and sentiment inference. Integrating these tasks into a single data pipeline requires achieving a unified input/output format at the information extraction level, maintaining semantic coherence at the logical level, and ensuring interpretability at the result level.

3. **System Design**
   
   This project proposes a scalable, multi-layered question-and-answer podcast understanding framework. The system adopts a modular structure design, enabling information sharing and hierarchical transmission while maintaining task boundaries. Ultimately, it presents a structured output format including a global summary, role summaries, and opinions/disagreements. This design balances interpretability and scalability.

---

## Proposed Datasets

### Primary Dataset
- **SPoRC: Speaker–Podcast Relationship Corpus** ([Hugging Face link](https://huggingface.co/datasets/blitt/SPoRC))  
  - Approximately 22 GB of English podcast transcripts.  
  - Contains both **episode-level** and **turn-level** files.  
  - Each turn includes text, inferred speaker role (`host`, `guest`, `cohost`, etc.), and episode metadata.

We will primarily use the **speaker-turn transcripts** for summarization and divergence detection. In addition, we will leverage **episode-level metadata** (titles, descriptions, and other attributes) to enrich our analysis. These fields provide high-level context, which can improve role-specific summaries and also serve as reference points for evaluation.

### Additional Data (Potential)
- We've discovered a podcast dataset that also features rounds. It includes a simple episode summary, the text of each round's speech, and the speaker's identity and name. Similar to SPoRC, we believe it's well-suited as an expansion dataset. Here's the link: https://github.com/zcgzcgzcg1/MediaSum
  - It is a large JSON file (4.45GB), with text quality slightly higher than the SPoRC dataset. We can process it using methods very similar to those used for the main dataset.

### Data Handling Plan
- We verified feasibility through prototype scripts that:
  - Convert `.jsonl.gz` → `.csv` using chunked reading (`pandas` + `tqdm`).
  - Filter only `host` and `guest` roles.
  - Clean text by removing empty or pure punctuational and ultra-short turns.
- Early experiments indicate that, with chunked reading and selective filtering, the dataset can be processed efficiently on standard machines (<16 GB RAM).

---

## Methods 

This project follows a **multi-stage pipeline** for role-aware podcast understanding, combining preprocessing, local divergence analysis, hierarchical summarization, and visualization. Some steps are already implemented (P2), while others are planned for P3.

---

### 1. Data Preprocessing (Implemented for P2)
- **Tools:** `pandas`, `gzip`, `csv`, `tqdm`
- Merge turn-level transcripts with episode metadata by `mp3url`.
- Clean and normalize speaker roles (retain only `host` and `guest`).
- Drop empty or extremely short utterances.
- Compute turn-level statistics (turn count, token length, per-episode balance).

This preprocessing pipeline ensures we can efficiently handle the large SPoRC dataset (20+ GB) and produces a clean dataset ready for further analysis.

---

### 2. Hierarchical Summarization Framework (Planned for P3)

This framework extends the baseline summarization to a **three-layer system** inspired by state-of-the-art works:

- Zou et al. (2021), *Topic-Oriented Spoken Dialogue Summarization*  
- Guan et al. (2024), *Role-Oriented Dialogue Summarization with CIAM*  

#### 2.1 Retrieval-Augmented Generation (RAG)
- Encode each episode segment with sentence embeddings (`all-MiniLM-L6-v2`) and store in a vector index (FAISS/Chroma).
- Retrieve top segments for a query $q$ using cosine similarity:

$$
\text{Retrieve}(q) = \arg\max_{u_i} \frac{e_q^\top e_i}{\lVert e_q \rVert \lVert e_i \rVert}
$$

- Retrieved content is used as context for summarization and conflict detection modules.

#### 2.2 Topic-Level Summarization
- Assign topic distributions $p(z|u_i)$ to each segment $u_i$.
- Compute saliency combining topic confidence and relevance to the query $q$:

$$
s(u_i) = \alpha \cdot \text{Rel}(u_i, q) + (1 - \alpha) \cdot \max_z p(z|u_i)
$$

$$
\text{Rel}(u_i, q) = \cos(f(u_i), f(q))
$$

- Top-scoring segments are passed to T5-small/BART/LLM for global topic summary $S_\text{topic}$.

#### 2.3 Role-Level Summarization
- Generate host/guest summaries conditioned on role token $r \in \{\text{HOST}, \text{GUEST}\}$:

$$
p(y | x, r) = \prod_t p(y_t | y<t, x, <r>)
$$

- CIAM contrastive learning enhances distinction between roles:

$$
L_\text{contra} = -\log \frac{\exp(\frac{\text{sim}(h_i^r, h_i^{+r})} {\tau)}}{\sum_j \exp(\frac{\text{sim}(h_i^r, h_j^{-¬r})} {\tau)}}
$$

- Outputs: $S_\text{host}$, $S_\text{guest}$

#### 2.4 Conflict Detection
- Semantic relations between $S_\text{host}$ and $S_\text{guest}$ classified using DeBERTa-v3-large-MNLI.
- Softmax probabilities for entailment, neutral, contradiction:

$$
p_k = \text{softmax}(W h + b)_k
$$

- Stance label determined via thresholds $\theta_e$, $\theta_c$:

```python
if p_entail >= θ_e:
    stance = "Complete Agreement"
elif p_contra >= θ_c:
    stance = "Complete Disagreement"
else:
    stance = "Partial Agreement / Neutral"
```

#### 2.5. Alternative Setup: Qwen 1.5B + LoRA

As an alternative, the pipeline can be replaced with a lightweight **Qwen 1.5B-Chat model fine-tuned via LoRA**,  
allowing end-to-end generation of role summaries and stance predictions under limited computational resources.

---

### 3. Visualization (Planned for P3)
- **Tools:** `matplotlib`, `seaborn`
- Visualize insights from preprocessing and analysis:
  - Role balance per episode (host vs guest speaking time).
  - Distribution and frequency of divergence points across episodes.
  - Cosine similarity curves between consecutive utterances.
  - Optional: visualize topic saliency scores and attention to understand summarization focus.

---

## Proposed Timeline

| Week                  | Planned Focus Area                    | Expected Output                                                                             |
|-----------------------|---------------------------------------|---------------------------------------------------------------------------------------------|
| **Week 45 (Current)** | Complete P2 deliverable               | Clean dataset subset, descriptive notebook, finalized README                                |
| **Week 46**           | RAG Retrieval Setup                   | Functional retriever (FAISS/Chroma), working query → top-K turns demo                       |
| **Week 47**           | Topic Summarization (AAAI 2021)       | Implement topic segmentation + saliency weighting; generate global summary S_topic          |
| **Week 48**           | Role Summarization (LREC-COLING 2024) | Role-conditioned summaries (S_host, S_guest); integrate with topic module                   |
| **Week 49**           | Conflict Detection (DeBERTa-MNLI)     | Add stance classification; merge retrieval + summarization + conflict into full QA pipeline |
| **Week 50**           | Optimization & Presentation           | Evaluate (ROUGE/BERTScore/stance accuracy), visualize results, prepare final demo/report    |
| **Week 51**           | Complete P3 deliverable               | Finalized notebook, visualizations, and report                                              |            

---

## Organization Within the Team

| Week                  | Milestone                                   |
|-----------------------|---------------------------------------------|
| **Week 45 (Current)** | Complete P2 deliverable                     | 
| **Week 47**           | Finalize Embedding and Divergence Detection | 
| **Week 49**           | Finalize Summarization & Baseline           |
| **Week 50**           | Complete Evaluation & Visualization         |
| **Week 51**           | Submit P3 deliverable                       | 


---

## Appendix

### Repo Organisation
This repository contains two files (for now):

- **`main.ipynb`** - Jupyter notebook containing all code related to the P2 hand-in.
- **`README.md`** - Detailed project proposal and documentation of methods, timeline, and team organization.


### Questions for TAs 
