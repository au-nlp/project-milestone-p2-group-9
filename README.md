
# Role-Aware Podcast Summarization and Divergence Detection

## Abstract
Podcasts are rich sources of spontaneous, multi-speaker dialogue that mix storytelling, debate, and information sharing. Yet, understanding long podcast episodes remains challenging for listeners and for downstream NLP applications.  
Our project aims to build a **role-aware podcast understanding(QA/RAG) system** that generates both **global** and **role-specific** summaries, while also detecting **divergent conversational segments**, meaning moments when the host and guest disagree or shift topics. Specifically:
1. When a topic is asked (e.g., "AI and education"), match the relevant TV series and output a summary related to the question.

2. When a specific question from a particular episode is asked (without specifying the episode, e.g., "Will AI exacerbate inequality in schools?"), provide a summary of the different characters' perspectives.

3. Of course, conflict detection (disagree, agree, neutral) is necessary during the summarization process.

We use the **SPoRC dataset**, a large-scale corpus of podcast transcripts with speaker-turn annotations, to model host-guest dynamics. Through embedding-based discourse analysis and transformer-based summarization models, we intend to create structured insights that reflect *who said what* and *where the conversation diverged*. This project demonstrates the feasibility of scalable dialogue understanding for complex multi-speaker audio data.

---

## Contributions and Novelty

1. **Role-Aware Summarization:**  
   Unlike standard podcast summarization, we explicitly distinguish between *host* and *guest* roles, producing summaries that capture each participant’s perspective on the different discussed topics.

2. **Divergence Detection:**  
   We introduce an interpretable method to detect and summarize **divergence points** - points of opinion conflict or topic shifts - by detecting sharp decreases in semantic similarity between speaker turns.

3. **Multi-Level Insights:**  
   The pipeline outputs three complementary views:
   - **Global episode summary** (overall themes)
   - **Role-specific summaries** (per speaker role)
   - **Divergence-focused mini-summaries** (debate or disagreement regions)

4. **Reproducible, Scalable Pipeline:**  
   We emphasize efficient handling of very large transcript data (20+ GB) through chunked preprocessing, embedding caching, and well-documented Jupyter workflows.

5. **Model Comparisons:**  
   To ensure robustness and interpretability, we plan to compare models of *similar scale* (e.g., T5-small vs. BART-base) rather than relying only on API-based LLMs.

---

## Contributions and Novelty(new)
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

### 1. Data Preprocessing
- **Tools:** `pandas`, `gzip`, `csv`, `tqdm`
- Merge turn-level transcripts with episode metadata by `mp3url`.  
- Clean and normalize speaker roles (retain only `host` and `guest`).  
- Drop empty or extremely short utterances.  
- Compute turn-level statistics (turn count, token length, per-episode balance).

### 2. Embedding and Divergence Detection
- **Model:** Sentence-BERT (`all-MiniLM-L6-v2`)  
- **Process:**
  1. Group transcripts by episode and order by turn.
  2. Compute cosine similarity between consecutive utterances.
  3. Identify “divergence points” when similarity drops below a dynamic threshold.
  4. (Potential) Validate using an NLI model (e.g., `microsoft/deberta-mnli`).

Mathematical sketch:  
$$
\text{divergence}(t_i, t_{i+1}) =
\begin{cases}
1, & \text{if } \cos(\mathbf{e_i}, \mathbf{e_{i+1}}) < \tau \\
0, & \text{otherwise}
\end{cases}
$$
where $(\tau)$ is a similarity threshold tuned on a validation subset.

### 3. Summarization
- **Baseline Models:** `t5-small`, `bart-base`  
- (Potential Extended Option): LLM API with few-shot prompts.  
- For each episode, we generate:
  - **Global summary:** input = all turns (truncated or concatenated with sampling).
  - **Role-specific summaries:** input = turns filtered by role.
  - **Divergence summary:** input = local context around divergence points.
- **Evaluation:** automatic (ROUGE, BERTScore) + qualitative inspection.
  - (Potential) Comparison with ASR based summaries.

### 4. Visualization
- **Tools:** `matplotlib`, `seaborn`  
- Visualize:
  - Role balance per episode.
  - Distribution of divergence points and divergence frequency.
  - Similarity curves over time.

---

## Methods(new)
This project adopts a three-layer hierarchical framework — **topic-level**, **role-level**, and **conflict-level** — inspired by two state-of-the-art works:  
- [1] Zou et al. (2021). *Topic-Oriented Spoken Dialogue Summarization for Customer Service with Saliency-Aware Topic Modeling.* AAAI 2021.  
- [2] Guan et al. (2024). *Improving Role-Oriented Dialogue Summarization with Interaction-Aware Contrastive Learning (CIAM).* LREC-COLING 2024.  
The bottom layer employs Microsoft’s *DeBERTa-v3-large-MNLI* model for stance inference.

---

### **1. Retrieval-Augmented Generation (RAG) Construction**

This study adopts a **Retrieval-Augmented Generation (RAG)** framework that integrates retrieval, summarization, and stance detection modules to enable question answering across multiple podcast episodes.  

The system first uses a sentence embedding model (e.g., `all-MiniLM-L6-v2`) to encode each episode segment into dense vectors and stores them in a vector index (FAISS / Chroma).  
When a user submits a query *q*, the system computes its embedding *e_q* and retrieves the most relevant segments *u_i* based on cosine similarity:

$$
\text{Retrieve}(q) = \arg\max_{u_i} \frac{e_q^\top e_i}{\lVert e_q \rVert \lVert e_i \rVert}
$$

The retrieved content is concatenated into a contextual passage *C(q)* and sequentially processed by the summarization and conflict detection modules:  

---

### **2. Topic-Oriented Summarization (AAAI 2021)**

At the top layer, the system generates a **topic-focused global summary**.  
Each dialogue segment *$u_i$* is assigned a topic distribution *p(z | u_i)* through topic modeling.  
Its saliency score combines relevance to the query and topic confidence:

s(u_i) = α * Rel(u_i, q) + (1 - α) * max_z p(z | u_i)
Rel(u_i, q) = cosine(f(u_i), f(q))

Segments with the highest saliency are concatenated and passed to a summarization model (e.g., T5/BART/LLM) to obtain the topic summary *S_topic*.

---

### **3. Role-Oriented Summarization (LREC-COLING 2024)**

At the mid layer, the system produces **role-specific summaries** using the CIAM framework.  
Given dialogue *x* and role tag *r* ∈ {HOST, GUEST}, generation is conditioned on the role token:

p(y | x, r) = ∏t p(y_t | y<t, x, <r>)

To enhance distinction between roles, CIAM introduces interaction-aware contrastive learning:

L_contra = -log( exp(sim(h_i^r, h_i^+r) / τ)
/ Σ_j exp(sim(h_i^r, h_j^-¬r) / τ) )

where *h_i^r* is the role representation, *sim()* denotes cosine similarity, and *τ* is the temperature coefficient.  
This produces two summaries: *S_host* and *S_guest*.

---

### **4. Conflict Detection (Microsoft DeBERTa-v3-large-MNLI)**

At the bottom layer, semantic relations between *S_host* and *S_guest* are classified by the NLI model.  
The model outputs three probabilities — entailment, neutral, and contradiction — via softmax:

p_k = softmax(W * h + b)_k

The final stance label is determined by thresholding these probabilities:

if p_entail >= θ_e: stance = "Complete Agreement"
elif p_contra >= θ_c: stance = "Complete Disagreement"
else: stance = "Partial Agreement / Neutral"

Thresholds θ_e, θ_c ∈ [0.6, 0.7] are empirically tuned to control boundary sensitivity.

---

### **5. Alternative Setup: Qwen 1.5B + LoRA**

As an alternative, the pipeline can be replaced with a lightweight **Qwen 1.5B-Chat model fine-tuned via LoRA**,  
allowing end-to-end generation of role summaries and stance predictions under limited computational resources.

---

## Methods (merged)

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

### 2. Embedding and Divergence Detection (Planned for P3)
- **Model:** Sentence-BERT (`all-MiniLM-L6-v2`)  
- **Process:**
  1. Group transcripts by episode and order by turn.
  2. Compute cosine similarity between consecutive utterances.
  3. Identify “divergence points” when similarity drops below a dynamic threshold.
  4. (Potential) Validate using an NLI model (e.g., `microsoft/deberta-mnli`).

Mathematical sketch:  
$$
\text{divergence}(t_i, t_{i+1}) =
\begin{cases}
1, & \text{if } \cos(\mathbf{e_i}, \mathbf{e_{i+1}}) < \tau \\
0, & \text{otherwise}
\end{cases}
$$
where $(\tau)$ is a similarity threshold tuned on a validation subset.

---

### 3. Hierarchical Summarization Framework (Planned for P3)

This framework extends the baseline summarization to a **three-layer system** inspired by state-of-the-art works:

- Zou et al. (2021), *Topic-Oriented Spoken Dialogue Summarization*  
- Guan et al. (2024), *Role-Oriented Dialogue Summarization with CIAM*  

#### 3.1 Retrieval-Augmented Generation (RAG)
- Encode each episode segment with sentence embeddings (`all-MiniLM-L6-v2`) and store in a vector index (FAISS/Chroma).
- Retrieve top segments for a query $q$ using cosine similarity:

$$
\text{Retrieve}(q) = \arg\max_{u_i} \frac{e_q^\top e_i}{\lVert e_q \rVert \lVert e_i \rVert}
$$

- Retrieved content is used as context for summarization and conflict detection modules.

#### 3.2 Topic-Level Summarization
- Assign topic distributions $p(z|u_i)$ to each segment $u_i$.
- Compute saliency combining topic confidence and relevance to the query $q$:

$$
s(u_i) = \alpha \cdot \text{Rel}(u_i, q) + (1 - \alpha) \cdot \max_z p(z|u_i)
$$

$$
\text{Rel}(u_i, q) = \cos(f(u_i), f(q))
$$

- Top-scoring segments are passed to T5/BART/LLM for global topic summary $S_\text{topic}$.

#### 3.3 Role-Level Summarization
- Generate host/guest summaries conditioned on role token $r \in \{\text{HOST}, \text{GUEST}\}$:

$$
p(y | x, r) = \prod_t p(y_t | y_{<t}, x, <r>)
$$

- CIAM contrastive learning enhances distinction between roles:

$$
L_\text{contra} = -\log \frac{\exp(\text{sim}(h_i^r, h_i^{+r}) / \tau)}{\sum_j \exp(\text{sim}(h_i^r, h_j^{-¬r}) / \tau)}
$$

- Outputs: $S_\text{host}$, $S_\text{guest}$

#### 3.4 Conflict Detection
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

#### 3.5. Alternative Setup: Qwen 1.5B + LoRA

As an alternative, the pipeline can be replaced with a lightweight **Qwen 1.5B-Chat model fine-tuned via LoRA**,  
allowing end-to-end generation of role summaries and stance predictions under limited computational resources.

---

### 4. Visualization (Planned for P3)
- **Tools:** `matplotlib`, `seaborn`
- Visualize insights from preprocessing and analysis:
  - Role balance per episode (host vs guest speaking time).
  - Distribution and frequency of divergence points across episodes.
  - Cosine similarity curves between consecutive utterances.
  - Optional: visualize topic saliency scores and attention to understand summarization focus.

---

## Proposed Timeline

| Week                  | Planned Focus Area               | Expected Output                                                                         |
|-----------------------|----------------------------------|-----------------------------------------------------------------------------------------|
| **Week 45 (Current)** | Complete P2 deliverable          | Clean dataset subset, descriptive notebook, finalized README                            |
| **Week 46**           | Embedding & Divergence Detection | Implement sentence embedding pipeline, prototype cosine similarity computation          |
| **Week 47**           | Embedding & Divergence Detection | Tune similarity thresholds, validate divergence detection results                       |
| **Week 48**           | Summarization & Baseline         | Implement T5-small and BART-base summarization on sample episodes                       |
| **Week 49**           | Summarization & Baseline         | Tune model parameters and compare summarization quality                                 |
| **Week 50**           | Evaluation & Visualization       | Evaluate summaries (ROUGE + qualitative), visualize divergence frequency & role balance |
| **Week 51**           | Complete P3 deliverable          | Finalized notebook, visualizations, and report                                          |

---

## Proposed Timeline(new)

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

## Proposed Timeline(new)

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

## Proposed Timeline(merged)

| Week                  | Planned Focus Area                            | Expected Output                                                                                      |
| --------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Week 45 (Current)** | Complete P2 deliverable                       | Clean dataset subset, descriptive notebook, finalized README                                         |
| **Week 46**           | Data Handling & Embedding Setup               | Implement sentence embedding pipeline (`all-MiniLM-L6-v2`), store embeddings in FAISS/Chroma         |
| **Week 47**           | Divergence Detection & Retrieval              | Prototype cosine similarity for divergence points, optional NLI validation; basic RAG retrieval demo |
| **Week 48**           | Topic-Level Summarization                     | Implement topic segmentation, saliency scoring, and generate global summary `S_topic`                |
| **Week 49**           | Role-Level Summarization & Conflict Detection | Generate host/guest summaries (`S_host`, `S_guest`) with CIAM, integrate conflict detection          |
| **Week 50**           | Evaluation, Visualization & Optimization      | Evaluate summaries (ROUGE/BERTScore), stance accuracy; visualize role balance, divergence, topics    |
| **Week 51**           | Complete P3 deliverable                       | Finalized notebook, visualizations, and report                                                       |

---

## Organization Within the Team

| Week                  | Milestone                                   |
|-----------------------|---------------------------------------------|
| **Week 45 (Current)** | Complete P2 deliverable                     | 
| **Week 47**           | Finalize Embedding and Divergence Detection | 
| **Week 49**           | Finalize Summarization & Baseline           |
| **Week 50**           | Complete Evaluation & Visualization         |
| **Week 51**           | Submit P3 deliverable                       | 

## Organizeation Within the Team (merged)
| Week                  | Milestone                                                                       |
| --------------------- | ------------------------------------------------------------------------------- |
| **Week 45 (Current)** | Complete P2 deliverable: preprocessing, descriptive stats, README               |
| **Week 46**           | Implement embedding pipeline and set up FAISS/Chroma vector store               |
| **Week 47**           | Prototype divergence detection and basic RAG retrieval                          |
| **Week 48**           | Implement topic-level summarization (global summary `S_topic`)                  |
| **Week 49**           | Implement role-level summarization (`S_host`, `S_guest`) and conflict detection |
| **Week 50**           | Evaluate summaries (ROUGE/BERTScore/stance accuracy) and create visualizations  |
| **Week 51**           | Complete P3 deliverable: finalize notebook, visualizations, and report          |

---

## Appendix

### Repo Organisation
This repository contains two files (for now):

- **`main.ipynb`** - Jupyter notebook containing all code related to the P2 hand-in.
- **`README.md`** - Detailed project proposal and documentation of methods, timeline, and team organization.


### Questions for TAs 
