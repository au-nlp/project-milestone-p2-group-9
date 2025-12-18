# Role-Aware Dialogue Summarization for Podcasts Using Two-Stage Extraction and Generation

## Abstract / Project Overview
Podcasts are a rapidly growing source of long-form spoken content featuring multiple speakers, informal language, and frequent topic shifts. These properties make automatic summarization challenging, especially when users are interested not only in a global overview but also in understanding how different speakers contribute to a conversation.
This project implements a role-aware two-stage podcast summarization system that generates both a global episode summary and role-specific summaries for hosts and guests.  
The system is trained without manual sentence-level extractive annotations by leveraging pseudo-labels derived from semantic similarity between utterances and reference summaries. Experiments are conducted on the SPoRC (Speaker–Podcast Relationship Corpus) dataset.  
Contrastive-inspired embedding alignment ensures distinction between speaker roles.

---

## Project Contributions and Novelty
1. **Role-Aware Summarization Pipeline**
   
    We design and implement a summarization framework that produces both global episode summaries and role-specific summaries (host/guest), adapting role-aware modeling to the podcast domain where speaker dynamics are central.

2. **Pseudo-Label Supervision for Extractive Training**

   We avoid costly manual extractive annotations by constructing pseudo-labels based on semantic similarity between utterances and reference summaries, enabling effective training of the extractor component on the SPoRC dataset.

3. **Practical Two-Stage Architecture for Dialogue Summarization**
   
    We integrate an extractive sentence selector with an abstractive BART-based generator tailored to long-form, multi-speaker podcast data, and demonstrate its effectiveness at scale.

4. **Topic-Conditioned Content Selection**
   
    By incorporating topic-level representations during extraction and refinement, we improve selection of content relevant to overall episode semantics, particularly in long and multi-topical podcasts.

5. **Comprehensive Evaluation on SPoRC**

    We empirically evaluate the system using ROUGE-L for global and role-level summaries across hundreds of episodes, demonstrating reproducible and scalable performance.

---

## Dataset
We use the **SPoRC dataset**, which contains speaker-annotated podcast transcripts with both turn-level dialogue and episode-level summaries.

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
- A lightweight neural extractor assigns an importance score to each utterance
- The extractor is trained using **pseudo-labels**, constructed by selecting the top-k utterances most semantically similar to the reference summary
- Sentence similarity is computed using **Sentence-BERT**

This process avoids the need for costly manual extractive annotations.

#### 2. Abstractive Refinement
- Selected utterances are concatenated
- A **BART-based abstractive generator** produces the final summary
- The same pipeline supports:
  - global summaries
  - host-only summaries
  - guest-only summaries

---

## Training
- The extractor is trained using **binary cross-entropy loss**
- Optimization is performed with **AdamW**
- Training is conducted on a subset of episodes for computational feasibility
- The abstractive refiner uses pretrained weights and is applied during inference

---

## Evaluation
We evaluate summary quality using **ROUGE-L F1**, comparing generated summaries against the corresponding reference summaries provided in SPoRC.

Evaluation is performed separately for:
- global summaries
- host summaries
- guest summaries

---

## Project Structure
``dataset_tds.py`` # Dataset loading and pseudo-label generation  
``model_tds.py`` # Two-stage summarization model  
``train_tds.py`` # Extractor training loop  
``inference_tds.py`` # Summary generation  
``evaluation.py`` # ROUGE-L evaluation  
``sproc_pipeline2.ipynb`` # End-to-end experiment notebook  

---

## Updates Since Milestone P2
Compared to the initial project proposal, the final implementation:
- Focuses exclusively on **role-aware summarization**
- Removes retrieval, conflict detection, and stance analysis components
- Replaces planned supervised extractive training with **pseudo-label supervision**
- Consolidates experiments into a stable two-stage summarization pipeline

These changes allowed us to produce a robust and reproducible system while remaining close to our original objectives.

---

## Team Contributions
**Hjalte V. Vinther:**


**:**


**:**




Development was carried out collaboratively with shared responsibility across components.

---

## References
- Zou et al. (2021). *Topic-Oriented Spoken Dialogue Summarization*
- Guan et al. (2024). *Role-Oriented Dialogue Summarization with Interaction-Aware Contrastive Learning*
- SPoRC 
- MediaSum. *Github used for inspiration*: https://github.com/zcgzcgzcg1/MediaSum
- Topic-Dialog-Summ. *Github used for inspiration*: https://github.com/RowitZou/topic-dialog-summ