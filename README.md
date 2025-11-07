

# Role-Aware Podcast Summarization and Divergence Detection

## Abstract
Podcasts are rich sources of multi-speaker dialogue, blending storytelling, debate, and information sharing. Understanding long episodes is challenging for both listeners and NLP systems.
Our project builds a role-aware podcast understanding system that generates global and role-specific summaries while detecting divergent conversational segments where speakers disagree or shift topics. Specifically, the system aims to:
1. Retrieve and summarize podcast content relevant to a user query.

2. Answer open-ended questions by summarizing different speakers’ perspectives.

3. Detect points of agreement or disagreement between speakers.

We use the SPoRC dataset, a large corpus of speaker-annotated podcast transcripts, and combine embedding-based retrieval with transformer-based summarization to produce structured insights reflecting who said what and where conversation diverged. This demonstrates scalable dialogue understanding for multi-speaker audio.

---

## Contributions and Novelty
- **Structured Dialogue Understanding:** Extends podcast summarization to include speaker stances and disagreements.  
- **Unified Pipeline:** Integrates topic identification, role-conditioned summarization, and stance detection in a single framework.  
- **Scalable Design:** Modular QA system producing global summaries, role summaries, and conflict insights, balancing interpretability and scalability.

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

---

## Methods 

We adopt a multi-stage pipeline combining preprocessing, retrieval, hierarchical summarization, conflict detection, and visualization.

---

### 1. Hierarchical Summarization Framework 

This framework extends the baseline summarization to a **three-layer system** inspired by state-of-the-art works:

- Zou et al. (2021), *Topic-Oriented Spoken Dialogue Summarization*  
- Guan et al. (2024), *Role-Oriented Dialogue Summarization with CIAM*  

#### 1.1 Retrieval-Augmented Generation (RAG)
- Encode each episode segment with sentence embeddings (`all-MiniLM-L6-v2`) and store in a vector index (FAISS/Chroma).
- Retrieve top segments for a query $q$ using cosine similarity:

$$
\text{Retrieve}(q) = \arg\max_{u_i} \frac{e_q^\top e_i}{\lVert e_q \rVert \lVert e_i \rVert}
$$

- Retrieved content is used as context for summarization and conflict detection modules.

#### 1.2 Topic-Level Summarization
- Assign topic distributions $p(z|u_i)$ to each segment $u_i$.
- Compute saliency combining topic confidence and relevance to the query $q$:

$$
s(u_i) = \alpha \cdot \text{Rel}(u_i, q) + (1 - \alpha) \cdot \max_z p(z|u_i)
$$

$$
\text{Rel}(u_i, q) = \cos(f(u_i), f(q))
$$

- Top-scoring segments are passed to T5-small/BART/LLM for global topic summary $S_\text{topic}$.

#### 1.3 Role-Level Summarization
- Generate host/guest summaries conditioned on role token $r \in \{\text{HOST}, \text{GUEST}\}$:

$$
p(y | x, r) = \prod_t p(y_t | y_{\le t}, x, r)
$$

- CIAM contrastive learning enhances distinction between roles:

$$
L_\text{contra} = -\log \frac{\exp(\text{sim}(h_i^r, h_i^{+r}) / \tau)}{\sum_j \exp(\\text{sim}(h_i^r, h_j^{-r}) / \tau)}
$$

- Outputs: $S_\text{host}$, $S_\text{guest}$

#### 1.4 Conflict Detection
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

#### 1.5. Alternative Setup: Qwen 1.5B + LoRA

As an alternative, the pipeline can be replaced with a lightweight **Qwen 1.5B-Chat model fine-tuned via LoRA**,  
allowing end-to-end generation of role summaries and stance predictions under limited computational resources.

---

### 2. Evaluation and Visualization
- **Automatic Metrics:** ROUGE, BERTScore for summaries; stance accuracy against human labels.  
- **Qualitative Inspection:** Assess coherence, role separation, and alignment with source dialogue.  
- **Visualizations:** Role dynamics, divergence/conflict points, and topic relevance using `matplotlib` and `seaborn`.


---

## Proposed Timeline

| Week                  | Planned Focus Area                    | Expected Output                                                                             |
|-----------------------|---------------------------------------|---------------------------------------------------------------------------------------------|
| **Week 45 (Current)** | Complete P2 deliverable               | Clean dataset subset, descriptive notebook, finalized README                                |
| **Week 46**           | RAG Retrieval Setup                   | Functional retriever (FAISS/Chroma), working query → top-K turns demo                       |
| **Week 47**           | Topic Summarization (AAAI 2021)       | Implement topic segmentation + saliency weighting; generate global summary S_topic          |
| **Week 48**           | Role Summarization (LREC-COLING 2024) | Role-conditioned summaries (S_host, S_guest); integrate with topic module                   |
| **Week 49**           | Conflict Detection (DeBERTa-MNLI)     | Add stance classification; merge retrieval + summarization + conflict into full QA pipeline |
| **Week 50**           | Evaluation & Optimization             | Evaluate (ROUGE/BERTScore/stance accuracy), visualize results, prepare final demo/report    |
| **Week 51**           | Complete P3 deliverable               | Finalized notebook, visualizations, and report                                              |            

---

## Organization Within the Team

To ensure smooth collaboration, we assign core responsibilities while maintaining weekly check-ins and code reviews:

| Week                  | Focus Area                          | Lead Member | 
|-----------------------|-------------------------------------|-------------|
| **Week 45 (Current)** | Complete P2 deliverable             | All         | 
| **Week 46**           | RAG Retrieval                       | All         | 
| **Week 47**           | Topic-Level Summarization           | All         | 
| **Week 48**           | Role-Level Summarization            | All         | 
| **Week 49**           | Conflict Detection                  | All         | 
| **Week 50**           | Evaluation & Optimization           | All         | 
| **Week 51**           | Final P3 Integration and Submission | All         | 

We currently haven't done any distributions to individual people. This could change in the future. 

---

## Appendix

### Repo Organisation
This repository contains two files (for now):

- **`main.ipynb`** - Jupyter notebook containing all code related to the P2 hand-in.
- **`README.md`** - Detailed project proposal and documentation of methods, timeline, and team organization.


### Questions for TAs 

None currently
