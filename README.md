# Stable Matching Experiments with LLMs

## Overview

This project evaluates how large language models (LLMs) perform on structured reasoning tasks based on the Stable Matching Problem.

We compare two primary model types:

- Basic model  
- Reasoning model  

across four tasks involving matching, detection, correction, and preference reasoning.

An extended evaluation is also performed using an advanced model (Gemini) to analyze how performance scales with increasing problem size.

The experiments use synthetic datasets generated under the Impartial Culture (IC) setting.

The goal is to assess:

- correctness and validity  
- stability of matchings  
- reasoning ability across tasks  
- performance differences between models  

---

## Tasks

### Task 1 — Generate Stable Matching

Given preference lists, the model generates a complete matching.

**Metrics:**
- Parsed count  
- Valid matching  
- Stable matching  
- Exact match with ground truth  
- Average blocking pairs  

---

### Task 2 — Detect Instability

Given a matching, the model determines whether it is stable.

**Metrics:**
- Correct predictions  
- Accuracy  

---

### Task 3 — Resolve Instability

Given an unstable matching, the model fixes it.

**Metrics:**
- Stable matching count  
- Exact match count  
- Average blocking pairs  

---

### Task 4 — Preference Reasoning

The model answers preference-based reasoning questions.

**Metrics:**
- Correct answers  
- Accuracy  

---

## Key Results and Insights

- Models achieve high validity but struggle with stability, indicating difficulty in enforcing global constraints  
- Stable matching generation performance is low (~30%), showing challenges in multi-step reasoning  
- Preference reasoning tasks perform better (up to ~70% accuracy), highlighting strength in local reasoning  
- The reasoning model does not consistently outperform the basic model, indicating reliability limitations  
- Gemini performs well on smaller datasets but degrades significantly on larger instances, highlighting scalability challenges  

---

## Project Structure

```text
project/
├── data/
│   ├── 5_ic_processed.csv
│   ├── 20_ic_processed.csv
│   └── 50_ic_processed.csv
│
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── task1_stable_matching.py
│   ├── task2_detect_instability.py
│   ├── task3_resolve_instability.py
│   ├── task4_preference_reasoning.py
│   ├── io_utils.py
│   └── reporting.py
│
├── notebooks/
│   ├── experiments.ipynb
│   └── gemini_stable_matching_evaluation.ipynb
│
├── results/
│   ├── summaries/
│   ├── tables/
│   └── figures/
│
└── README.md

## Authors

Dana Aljamal  
Shahida Batool  
Nimrah Adam  
