# Role-Aware Dialogue Summarization for Podcasts Using Two-Stage Extraction and Generation

## Abstract / Project Overview
Podcasts are a rapidly growing source of long-form spoken content featuring multiple speakers, informal language, 
and frequent topic shifts. These properties make automatic summarization challenging, especially when users are 
interested not only in a global overview but also in understanding how different speakers contribute to a conversation. <br>
This project implements a role-aware two-stage podcast summarization system that generates both a global episode 
summary and role-specific summaries for hosts and guests. The system is trained without manual sentence-level 
extractive annotations by leveraging pseudo-labels derived from semantic similarity between utterances and 
reference summaries. Experiments are conducted on the SPoRC (Speaker–Podcast Relationship Corpus) dataset.  

---

## Project Contributions and Novelty
1. **Role-Aware Summarization Pipeline**
   
    We design and implement a summarization framework that produces both global episode summaries and role-specific 
summaries (host/guest), explicitly modeling speaker roles in multi-party podcast dialogue.

2. **Pseudo-Label Supervision for Extractive Training**

   We avoid costly manual extractive annotations by constructing pseudo-labels by selecting utterances that are most 
semantically similar to reference summaries. This enables effective training of the extractor using only episode-level summaries.

3. **Two-Stage Architecture for Dialogue Summarization**
   
    We combine a lightweight neural extractor for utterance selection with a BART-based abstractive generator, 
forming a practical and scalable approach to long-form dialogue summarization.

4. **Role-Controlled Abstractive Generation**
   
    Role control tokens are used to guide the abstractive model toward generating global, host-specific, or 
guest-specific summaries within a unified architecture.

5. **Reproducible Evaluation on SPoRC**

    We empirically evaluate the system using ROUGE-L for global and role-level summaries across hundreds of episodes, 
demonstrating reproducible and scalable performance.

---

## Dataset
We use the **SPoRC dataset**, which contains speaker-annotated podcast transcripts with both turn-level dialogue and 
episode-level summaries.

- Turn-level CSV:
  - utterance text
  - speaker role (e.g., host, guest)
  - episode and turn indices
- Episode-level summaries:
  - global summary
  - host summary
  - guest summary

Episodes without a global summary are excluded from training, and role-specific summaries are used only when available.

---

## Method

### Two-Stage Summarization Pipeline
Our system follows a **two-stage extractive–abstractive framework**:

#### 1. Extractive Utterance Selection
- Each utterance is encoded using a **pretrained BART encoder**
- A lightweight neural extractor assigns an importance score to each utterance within an episode
- The extractor is trained using **pseudo-labels**, constructed by selecting the top-k utterances most semantically 
similar to the reference summary
- Sentence similarity is computed using **Sentence-BERT**

This approach enables effective extractive training without manual annotation.

#### 2. Abstractive Refinement
- Selected utterances are concatenated
- A **BART-based abstractive generator** produces the final summary
- The same pipeline supports:
  - global summaries
  - host-only summaries
  - guest-only summaries

---

## Training
- Extractor training:
  - trained using **binary cross-entropy loss**
  - optimization is performed with **AdamW**
  - frozen utterance encoder used
- Refiner training:
  - supervised sequence-to-sequence learning
  - reference summaries used as silver targets
- Training is conducted on a subset of episodes for computational feasibility

Together, this training setup enables efficient learning of both sentence selection and role-aware summary generation 
without requiring manual extractive annotations.

---

## Evaluation

- Summary quality is evaluated using **ROUGE-L F1**
- Generated summaries are compared against fixed LLM-generated silver reference summaries
- Evaluation is performed separately for:
  - global summaries
  - host summaries
  - guest summaries

This separation allows us to assess both overall summarization quality and the models ability to capture role-specific perspectives.

---

## Project Structure
modules <br>
|-> ``dataset_tds.py`` — SPoRC dataset loading, episode grouping, and role-aware pseudo-label generation <br>
|-> ``inference_tds.py`` — Episode-level summary generation (extractive selection + abstractive BART refinement) <br>
|-> ``model_tds.py`` — Two-stage summarization model with extractive utterance selection and BART-based abstractive refinement <br>
|-> ``old_README.md`` — The README.md file from the P2 milestone <br>
|-> ``train_tds.py`` — # Train the TDS two-stage model (extractor + abstractive refiner) using pseudo-labels and SATM topic vectors <br>
|-> ``train_satm.py`` — Train SATM topic model for producing episode-level topic vectors used in TDS extraction <br>
``main.ipynb`` — The main notebook used for the project. A merge of the other files, to collect the main logic in one place <br>
``preprocessing.ipynb`` — The preprocessing part of the code. Part 1-8 in the main.ipynb file <br>
``README.md`` — README file for an overview of the code and project, included team contributions <br>
``report.pdf`` — The project report <br>
``sporc_fake_summary.ipynb`` — End-to-end experiment notebook, built around a local LLM (Qwen 2.5). Part 9-12 in the main.ipynb file <br>
``sporc_full_summaries_7b.jsonl`` — Line by line JSON file containing full episodic summaries <br>
``sproc_pipeline2-final.ipynb`` — End-to-end experiment notebook, with both results and analysis. Part 13-25 in the main.ipynb file <br>

---

## Comparison to the plan of the Milestone P2
Since the P2 proposal, the project has undergone several refinements to better align the system design with practical constraints and implementation feasibility.

Compared to the original plan, the final system:
- Narrows the summarization pipeline to a **two-stage extractive–abstractive framework**, removing explicit topic modeling and contrastive learning components
- Replaces assumed supervised extractive training with **weak supervision via LLM-generated silver summaries**, enabling scalable training without sentence-level annotations
- Retains **role-aware summarization** as a core objective, producing global, host-specific, and guest-specific summaries
- Implements **viewpoint conflict detection** using a pretrained NLI model over role-specific summaries, instead of stance classification against human labels
- Simplifies retrieval-augmented generation into a **retrieval-based QA interface** over pre-generated summaries and conflict evidence

These changes allowed the project to transition from a broad exploratory design to a focused, reproducible system that 
preserves the original goals of role-aware understanding and speaker divergence analysis, while remaining feasible within the project scope.

---

## Team Contributions
**Hjalte V. Vinther:** <br>
Literature searching and reading, conducting research, setting up the main.ipynb file, assisting with coding throughout the different parts of the system, writing and proofreading the README.md file, report writing and polishing/proofreading.

**Jiqiang Dong:** <br>
Finding and reading reference papers, conducting research, mainly generating the pseudo-summary dataset, running large-scale training on the full dataset, and contributing to report writing.

**Chuanrui Tang:** <br>
Finding and reading reference papers, researching and constructing ideas, writing the main code of the model algorithm and conducting small-scale training, and writing the main body of the report.



Development was carried out collaboratively with shared responsibility across components.

---

## References
- Zou et al. (2021). *Topic-Oriented Spoken Dialogue Summarization*
- Guan et al. (2024). *Role-Oriented Dialogue Summarization with Interaction-Aware Contrastive Learning*
- SPoRC 
- MediaSum. *Github used for inspiration*: https://github.com/zcgzcgzcg1/MediaSum
- Topic-Dialog-Summ. *Github used for inspiration*: https://github.com/RowitZou/topic-dialog-summ
- We uploaded the cleaned data, the generated pseudo-summary dataset, and the resulting test results. *https://drive.google.com/drive/folders/1nWZvH3YusY4N4QbCE3cbcXdEg04FLClT*
