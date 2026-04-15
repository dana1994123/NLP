# Stable Matching Experiments with LLMs

## Overview

This project evaluates how large language models (LLMs) perform on a set of matching and reasoning tasks derived from the Stable Matching Problem.

We compare two model types:

* **Basic model**
* **Reasoning model**

Each model is evaluated across **4 tasks**, using:

* **10 instances**
* **20 instances**

The goal is to measure:

* correctness
* stability
* reasoning ability
* performance differences between models

---

## Tasks

### Task 1 — Generate Stable Matching

Given preference lists, the model generates a full matching.

**Metrics:**

* Parsed count
* Valid matching
* Stable matching
* Exact match with ground truth
* Average blocking pairs

---

### Task 2 — Detect Instability

Given a matching, the model determines whether it is stable.

**Metrics:**

* Correct predictions
* Accuracy

---

### Task 3 — Resolve Instability

Given an unstable matching, the model fixes it.

**Metrics:**

* Stable matching count
* Exact match count
* Average blocking pairs

---

### Task 4 — Preference Reasoning

The model answers preference-based reasoning questions.

**Metrics:**

* Correct answers
* Accuracy

---

## Project Structure

```text
project/
├── data/
│   └── 5_ic_processed.csv
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
│   └── final_experiments.ipynb
│
├── results/
│   ├── summaries/
│   ├── tables/
│   └── figures/
│
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install langchain-groq pandas matplotlib
```

---

### 2. API Key

You will be prompted to enter your **Groq API key** when running the notebook.

---

## Running the Experiments

Open the notebook:

```bash
jupyter notebook
```

Then run:

```text
notebooks/final_experiments.ipynb
```

---

### Workflow

1. Run each task (Task 1–4)
2. Run for:

   * 10 instances
   * 20 instances
3. Save summaries to:

```text
results/summaries/
```

---

## Saved Outputs

### Summaries (JSON)

Each experiment is saved as:

```text
results/summaries/taskX_model_instances.json
```

Example:

```text
task1_basic_10.json
task2_reasoning_20.json
```

---

### Final Table

```text
results/tables/final_summary_table.csv
```

Contains all tasks, models, and instance sizes.

---

### Grouped Bar Chart

```text
results/figures/grouped_bar_chart.png
```

Shows model performance across tasks.

---

## Final Results

### Summary Table

* Combines all experiments into a single DataFrame
* Used for analysis and reporting

### Chart

* Compares:

  * Basic vs Reasoning models
  * Across all tasks
  * Across instance sizes (10 vs 20)

---

## Key Insights (Example)

* Reasoning models typically perform better on:

  * Task 2 (instability detection)
  * Task 4 (preference reasoning)

* Basic models can still perform competitively on:

  * Task 1 (matching generation)

* Increasing instances (10 → 20):

  * Improves reliability of results
  * Highlights consistency differences between models

---

## Notes

* All reusable code is placed in `src/`

* The notebook is used only for:

  * running experiments
  * loading results
  * visualization

* Results are saved to avoid recomputation

---

## Next Steps

* Add more datasets
* Try different LLMs
* Improve prompt design
* Normalize metrics for better comparison

---

## Author

Dana Aljamal
Shahida Batool
Nimrah Adam
