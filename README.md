
# Role-Aware Podcast Summarization and Divergence Detection

## Abstract
Podcasts are rich sources of spontaneous, multi-speaker dialogue that mix storytelling, debate, and information sharing. Yet, understanding long podcast episodes remains challenging for listeners and for downstream NLP applications.  
Our project aims to build a **role-aware podcast understanding system** that generates both **global** and **role-specific** summaries, while also detecting **divergent conversational segments**, meaning moments when the host and guest disagree or shift topics.  
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

## Proposed Datasets

### Primary Dataset
- **SPoRC: Speaker–Podcast Relationship Corpus** ([Hugging Face link](https://huggingface.co/datasets/blitt/SPoRC))  
  - Approximately 22 GB of English podcast transcripts.  
  - Contains both **episode-level** and **turn-level** files.  
  - Each turn includes text, inferred speaker role (`host`, `guest`, `cohost`, etc.), and episode metadata.

We will primarily use the **speaker-turn transcripts** for summarization and divergence detection. In addition, we will leverage **episode-level metadata** (titles, descriptions, and other attributes) to enrich our analysis. These fields provide high-level context, which can improve role-specific summaries and also serve as reference points for evaluation.

### Additional Data (Potential)
- If time permits, we might include **ASR-based summaries** (Automatic Speech Recognition) from open podcast APIs (e.g., Podscribe or ListenNotes) to compare our model to that of a generic automated summary.

### Data Handling Plan
- We verified feasibility through prototype scripts that:
  - Convert `.jsonl.gz` → `.csv` using chunked reading (`pandas` + `tqdm`).
  - Filter only `host` and `guest` roles.
  - Clean text by removing empty or ultra-short turns (<10 characters).
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

## Proposed Timeline

| Week              | Milestone | Expected Output |
|-------------------|------------|-----------------|
| **Week 45 (Now)** | Complete P2 deliverable | Clean dataset subset, descriptive notebook, finalized README |
| **Week 46**       |  |  |
| **Week x**        |  |  |
| **Week y**        |  |  |
| **Week 5**        |  |  |

---

## Organization Within the Team


| Milestone         | Milestone | Expected Output |
|-------------------|------------|-----------------|
| **Week 45 (Now)** | Complete P2 deliverable | Clean dataset subset, descriptive notebook, finalized README |
| **Week 46**       |  |  |
| **Week x**        |  |  |
| **Week y**        |  |  |
| **Week 5**        |  |  |

---

## Appendix

### Repo Organisation


### Questions for TAs 