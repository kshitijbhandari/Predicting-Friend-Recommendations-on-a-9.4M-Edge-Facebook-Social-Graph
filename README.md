# Social Network Graph Link Prediction — Facebook Challenge

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approach](#approach)
- [Feature Engineering](#feature-engineering)
- [Model & Results](#model--results)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [References](#references)

---

## Problem Statement

Given a **directed social graph** (from Facebook's recruiting challenge on Kaggle), the task is to **predict missing links** — i.e., recommend users that a given user is likely to follow.

- **Dataset Source:** [Facebook Recruiting Challenge — Kaggle](https://www.kaggle.com/c/FacebookRecruiting)
- **Task Type:** Binary Classification (link exists = 1, link does not exist = 0)
- **Performance Metric:** F1 Score (both precision and recall are important) + Confusion Matrix
- **No low-latency constraint** — we can afford heavier graph-based feature computation offline.

---

## Dataset

The raw data contains **9,437,519 directed edges** across **1,862,220 unique nodes** (users).

| Column | Description |
|---|---|
| `source_node` | User who is following |
| `destination_node` | User being followed |

### Key Statistics

| Metric | Value |
|---|---|
| Total edges | 9,437,519 |
| Unique nodes | 1,862,220 |
| Average in-degree (followers) | 5.07 |
| Average out-degree (following) | 5.07 |
| Max followers | 552 |
| Max following | 1,566 |
| 99th percentile followers | 40 |
| Users with zero following | 14.74% |
| Users with zero followers | 10.10% |
| Weakly connected components | 45,558 |

> **Note:** No isolated nodes exist — every user has at least one incoming or outgoing edge.

---

## Approach

The link prediction problem is reframed as a **supervised binary classification** problem.

### 1. Negative Sampling
Since the dataset only contains existing edges (positive samples), we need to generate **negative samples** (non-existent edges). Negative edges are generated randomly with the constraint that the **shortest path between the two nodes must be greater than 2** — this ensures negatives are not trivially separated.

- Positive samples: 9,437,519 (existing edges)
- Negative samples: 9,437,519 (generated missing edges, shortest path > 2)

### 2. Train-Test Split
Positive and negative samples are split separately to maintain balance:

| Split | Positive | Negative | Total |
|---|---|---|---|
| Train (80%) | 7,550,015 | 7,550,015 | 15,100,030 |
| Test (20%) | 1,887,504 | 1,887,504 | 3,775,008 |

> **Cold Start Issue:** 7.12% of test nodes are not present in the training graph.

The **training graph** is built using only the 80% positive edges. This graph is then used for all feature extraction (to prevent data leakage).

### 3. Feature Extraction
Rich graph-based features are extracted for each (source, destination) node pair in 4 stages.

### 4. Model Training
An **XGBoost classifier** is trained and evaluated using F1 score. Hyperparameters are tuned based on minimising the overfitting gap (train F1 − test F1).

---

## Feature Engineering

**Total Features: 59**

### Stage 1 — Basic Network Features (6 features)
| Feature | Description |
|---|---|
| `num_followers_s` | Number of followers of source node |
| `num_followees_s` | Number of people source node follows |
| `num_followers_d` | Number of followers of destination node |
| `num_followees_d` | Number of people destination node follows |
| `inter_followers` | Common followers between source and destination |
| `inter_followees` | Common followees between source and destination |

### Stage 2 — Graph Properties (4 features)
| Feature | Description |
|---|---|
| `adar_index` | Adamic/Adar index — weighted sum of inverse log degrees of common neighbors |
| `follows_back` | Binary: does destination follow source back? |
| `same_comp` | Binary: do source and destination belong to the same weakly connected component? |
| `shortest_path` | Shortest path length between nodes (edge removed if direct edge exists; -1 if unreachable) |

> **Adamic/Adar formula:** $A(x,y) = \sum_{u \in N(x) \cap N(y)} \frac{1}{\log|N(u)|}$

### Stage 3 — Weight and Centrality Features (14 features)

**Edge Weight Features** (inversely proportional to node degree — popular nodes carry less signal):

$$W = \frac{1}{\sqrt{1 + |X|}}$$

| Feature | Description |
|---|---|
| `weight_in` | Weight based on in-degree of destination |
| `weight_out` | Weight based on out-degree of source |
| `weight_f1` | `weight_in + weight_out` |
| `weight_f2` | `weight_in × weight_out` |
| `weight_f3` | `2×weight_in + weight_out` |
| `weight_f4` | `weight_in + 2×weight_out` |

**Centrality Features:**

| Feature | Description |
|---|---|
| `page_rank_s` | PageRank score of source node |
| `page_rank_d` | PageRank score of destination node |
| `katz_s` | Katz centrality of source (α=0.005, β=1) |
| `katz_d` | Katz centrality of destination |
| `hubs_s` | HITS hubs score of source |
| `hubs_d` | HITS hubs score of destination |
| `authorities_s` | HITS authorities score of source |
| `authorities_d` | HITS authorities score of destination |

> **PageRank** ranks nodes by incoming link structure. **Katz centrality** generalises eigenvector centrality by considering all paths. **HITS** separates nodes into hubs (good outgoing links) and authorities (good incoming links).

### Stage 4 — SVD Features (24 features)
SVD is applied to the adjacency matrix (1,780,722 × 1,780,722) to extract a **6-dimensional latent vector** for each node.

| Feature Group | Description |
|---|---|
| `svd_u_s_1` to `svd_u_s_6` | Left singular vectors (U matrix) for source node |
| `svd_u_d_1` to `svd_u_d_6` | Left singular vectors (U matrix) for destination node |
| `svd_v_s_1` to `svd_v_s_6` | Right singular vectors (V matrix) for source node |
| `svd_v_d_1` to `svd_v_d_6` | Right singular vectors (V matrix) for destination node |

### Additional Engineered Features (5 features)
| Feature | Description |
|---|---|
| `num_followers_d` | Re-added from full graph (was missing in initial stage) |
| `preferential_attachment_followers` | `num_followers_s × num_followers_d` |
| `preferential_attachment_followees` | `num_followees_s × num_followees_d` |
| `svd_dot_u` | Dot product of source and destination U vectors |
| `svd_dot_v` | Dot product of source and destination V vectors |

---

## Model & Results

### XGBoost Hyperparameter Tuning
Grid search over 36 configurations. Best hyperparameters selected by **minimum (train F1 − test F1)** gap to minimise overfitting.

| Parameter | Search Space | Best Value |
|---|---|---|
| `n_estimators` | 30, 60, 100, 200, 300, 400 | **30** |
| `max_depth` | 3, 7, 12 | **7** |
| `subsample` | 0.4, 0.7 | **0.7** |

> Column sampling was deliberately excluded — with a limited number of features, we want all sub-models to learn from the most important features.

### Performance

| Metric | Value |
|---|---|
| **Train F1 Score** | **0.9802** |
| **Test F1 Score** | **0.9325** |
| Overfitting Gap | 0.0477 |
| ROC-AUC (without new features) | 0.93 |
| ROC-AUC (with preferential attachment + SVD dot) | **0.94** |

### Confusion Matrix Analysis
- Model tends to classify more test points as class 0 (non-link) rather than class 1 (link)
- Precision for class 0 decreases on test data; class 1 precision stays stable
- Recall for class 1 drops significantly on test data
- → Model is biased towards predicting the negative class on unseen data

### Feature Importance (Top Findings)
1. **`follows_back`** — by far the most important feature; whether the destination follows the source back is the strongest predictor of a link
2. **`preferential_attachment_followers`** — significant; popular users are more likely to follow each other
3. Weight and centrality features — moderate importance
4. **`svd_dot_u` / `svd_dot_v`** — not in the top 25 most important features; SVD dot products add limited value

---

## Project Structure

```
├── 1_EDA_and_Data_Preparation.ipynb        # EDA, negative sampling, train-test split
├── 2_Feature_Engineering.ipynb             # Feature extraction (all 4 stages)
├── 3_Model_Training_and_Evaluation.ipynb   # XGBoost training, tuning, evaluation
├── requirements.txt                        # Python dependencies
├── README.md
│
└── data/                                   # (not included — download from Kaggle)
    ├── train.csv                           # Raw Facebook graph edges
    ├── after_eda/
    │   ├── train_woheader.csv
    │   ├── train_pos_after_eda.csv
    │   ├── test_pos_after_eda.csv
    │   ├── train_neg_after_eda.csv
    │   ├── test_neg_after_eda.csv
    │   ├── train_after_eda.csv
    │   ├── test_after_eda.csv
    │   └── missing_edges_final.p
    ├── train_y.csv
    ├── test_y.csv
    └── fea_sample/
        ├── katz.p
        ├── hits.p
        ├── storage_sample_stage1.h5
        ├── storage_sample_stage2.h5
        ├── storage_sample_stage3.h5
        └── storage_sample_stage4.h5
```

---

## How to Run

### Prerequisites
1. Download `train.csv` from [Kaggle — Facebook Recruiting Challenge](https://www.kaggle.com/c/FacebookRecruiting)
2. Place it in the `data/` directory
3. Install dependencies: `pip install -r requirements.txt`

### Execution Order
Run the notebooks **in order**:

```
1_EDA_and_Data_Preparation.ipynb
    → Performs EDA on the graph
    → Generates negative samples
    → Creates train/test split CSVs in data/after_eda/

2_Feature_Engineering.ipynb
    → Builds training graph from positive training edges
    → Extracts all 59 features in 4 stages
    → Saves intermediate results to HDF5 files (storage_sample_stageX.h5)

3_Model_Training_and_Evaluation.ipynb
    → Loads features from storage_sample_stage4.h5
    → Adds additional engineered features
    → Grid search over XGBoost hyperparameters
    → Evaluates best model — confusion matrix, ROC-AUC, feature importance
```

> **Note:** Feature computation (especially Katz centrality, HITS, SVD on the full 1.8M node graph) is computationally expensive and may take hours. Intermediate results are cached to disk automatically using `os.path.isfile()` checks throughout the notebooks.

---

## Dependencies

```
pandas
numpy
networkx
matplotlib
seaborn
scikit-learn
xgboost
scipy
tqdm
tables          # for HDF5 support via pandas
plotly
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## References

- [Kleinberg, J. (2003) — Link Prediction in Networks (Cornell)](https://www.cs.cornell.edu/home/kleinber/link-pred.pdf)
- [Lichtenwalter et al. (2010) — New Perspectives and Methods in Link Prediction](https://www3.nd.edu/~dial/publications/lichtenwalter2010new.pdf)
- [Cukierski et al. — Graph-based Features for Supervised Link Prediction](https://kaggle2.blob.core.windows.net/forum-message-attachments/2594/supervised_link_prediction.pdf)
- [Applied AI Course — Link Prediction (YouTube)](https://www.youtube.com/watch?v=2M77Hgy17cg)
- [Adamic/Adar Index](https://en.wikipedia.org/wiki/Adamic%E2%80%93Adar_index)
- [PageRank](https://en.wikipedia.org/wiki/PageRank)
- [Katz Centrality](https://en.wikipedia.org/wiki/Katz_centrality)
- [HITS Algorithm](https://en.wikipedia.org/wiki/HITS_algorithm)
