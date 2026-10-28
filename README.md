# Face Spoofing Attack Detection in Cyberphysical Space
### Hybrid Multimodal Fusion: TSBTC (Machine Learning) + Fine-Tuned VGG16 (Deep Learning)
**B.Tech Final Year Capstone Project**  
*Evaluated on the NUAA Face Anti-Spoofing Benchmark Dataset (12,614 Images)*

---

## 📌 1. Project Overview

Biometric authentication systems deployed in cyberphysical environments are highly vulnerable to presentation attacks (spoofing), such as printed photograph attacks, digital replay attacks, and mask presentations. 

This project implements an end-to-end framework combining **Handcrafted Color-Texture Descriptors (TSBTC)** and **Deep Convolutional Representations (VGG16)** via **Early Feature Concatenation** and **Multi-Family Ensemble Learning** under rigorous **Stratified 10-Fold Cross-Validation**.

### The 3 Core Stages:
1. **Stage 1 — Handcrafted ML Baseline (TSBTC)**: Extracts Thepade's Sorted Block Truncation Coding across RGB color planes ($3n$ features for 11 variations of $n\_parts$) and evaluates 19 model configurations across 4 machine learning families and multi-level ensembles.
2. **Stage 2 — Deep Learning Baseline (Fine-Tuned VGG16)**: End-to-end fine-tuning of a 134.27M-parameter VGG16 architecture, extracting 1,048-dimensional continuous deep representations pooled from the penultimate dense layer (`fc2`).
3. **Stage 3 — Feature Concatenation & Multimodal Fusion**: Early fusion concatenating normalized TSBTC ($[0, 1]$) and continuous VGG16 ($[0, 1]$) representations ($1048 + 3n$ dimensions), evaluated with 10-fold CV across all 19 classifiers.

---

## 🔬 2. Methodology & Architecture

```
                                  [ Input Face Image (224 x 224) ]
                                            /          \
                     +---------------------+            +-----------------------+
                     |                                                          |
             [ TSBTC Feature Extraction ]                              [ Fine-Tuned VGG16 Backbone ]
                     |                                                          |
       - Sort R, G, B plane pixels (Eq. 1)                             - 13 Conv Layers + MaxPool Blocks
       - Partition into n contiguous parts (n_parts)                   - Penultimate Dense Layer fc2 (4096-D)
       - Part mean intensity features (Eq. 2)                          - Adaptive 1D AvgPool -> 1048-D
                     |                                                          |
             Raw 3n Vector                                             Min-Max Scaling per Image
                     |                                                          |
       Scaled by (1 / 255.0) -> [0.0, 1.0]                     Continuous Activations in [0.0, 1.0]
                     \                                                          /
                      \                                                        /
                       +-------------------> [ Early Fusion ] <---------------+
                                                    |
                                      Concatenated Feature Vector
                                  Dimension: 1048 + 3n (1192 at n=48)
                                  Scale: Both features strictly in [0, 1]
                                                    |
                                 [ Stratified 10-Fold Cross-Validation ]
                                                    |
                   +--------------------------------+--------------------------------+
                   |                                |                                |
            [ Tree Family ]                 [ Boosting Family ]             [ Distance & Kernel ]
         - Decision Tree                 - Hist Gradient Boosting        - K-Nearest Neighbors
         - Random Forest                 - XGBoost                       - SVM (RBF Kernel)
         - Extra Trees                   - LightGBM                      - Linear SVM
         - Intra-Family Voting           - CatBoost                      - Intra-Family Voting
                                         - Intra-Family Voting
                                                    |
                                       [ Probabilistic & Bayes ]
                                     - Gaussian Naive Bayes
                                     - Quadratic Discriminant (SafeQDA)
                                     - Bernoulli Naive Bayes
                                     - Intra-Family Voting
                                                    |
                                 +------------------+------------------+
                                 |                                     |
                 [ Inter-Family Voting Ensemble ]      [ Inter-Family Stacking Ensemble ]
                   (Consensus of the 4 families)         (Logistic Regression Meta-Classifier)
                                                    |
                                       Final Decision: Real vs. Spoof
```

---

## 📊 3. Key Findings & Experimental Results

All experiments were evaluated on the complete **12,614 images** of the NUAA dataset (5,105 genuine client images, 7,509 imposter spoof attacks).

### 🏆 Executive Summary Table (`Final_Summary`)

| Stage | Classifier | Accuracy% | Precision% | Recall% | F1% | ROC-AUC | FAR% | FRR% | TP | TN | FP | FN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ML Baseline (TSBTC Best $n=48$)** | Inter-Family Stacking Ensemble | **99.42%** | 99.72% | 98.84% | 99.28% | 0.9959 | 0.19% | 1.16% | 5,046 | 7,495 | 14 | 59 |
| **DL Baseline (VGG16)** | Fine-Tuned VGG16 (Standalone) | **99.98%** | 99.96% | 99.98% | 99.97% | 1.0000 | 0.03% | 0.02% | 5,104 | 7,507 | 2 | 1 |
| **Fused (TSBTC_n48 + VGG16)** | **K-Nearest Neighbors** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **0.00%** | **0.00%** | **5,105** | **7,509** | **0** | **0** |
| **Fused (TSBTC_n48 + VGG16)** | **XGBoost** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **0.00%** | **0.00%** | **5,105** | **7,509** | **0** | **0** |
| **Fused (TSBTC_n48 + VGG16)** | **Hist Gradient Boosting** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **0.00%** | **0.00%** | **5,105** | **7,509** | **0** | **0** |
| **Fused (TSBTC_n48 + VGG16)** | **Inter-Family Stacking Ensemble** | **99.99%** | **100.00%** | **99.98%** | **99.99%** | **1.0000** | **0.00%** | **0.02%** | **5,104** | **7,509** | **0** | **1** |

---

### 📈 Table III: Ranking of All 19 Classifiers on Fused Features

Under 10-fold cross-validation on 1,192-dimensional fused vectors:

| Rank | Model Family | Classifier / Ensemble | Accuracy% | Precision% | Recall% | F1% | FAR% | FRR% |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Distance & Kernel | **K-Nearest Neighbors** | **100.00%** | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| **2** | Boosting | **XGBoost** | **100.00%** | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| **3** | Boosting | **Hist Gradient Boosting** | **100.00%** | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| **4** | Distance & Kernel | **Intra-Family Voting Ensemble** | **100.00%** | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| **5** | Distance & Kernel | **Linear SVM** | **100.00%** | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| **6** | Distance & Kernel | **SVM (RBF Kernel)** | **100.00%** | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| **7** | Tree | Random Forest | **99.99%** | 100.00% | 99.98% | 99.99% | 0.00% | 0.02% |
| **8** | Tree | Extra Trees | **99.99%** | 100.00% | 99.98% | 99.99% | 0.00% | 0.02% |
| **9** | Ensemble Learning | **Inter-Family Stacking Ensemble** | **99.99%** | 100.00% | 99.98% | 99.99% | 0.00% | 0.02% |
| **10** | Ensemble Learning | Inter-Family Voting Ensemble | **99.99%** | 100.00% | 99.98% | 99.99% | 0.00% | 0.02% |
| **11** | Boosting | Intra-Family Voting Ensemble | **99.99%** | 100.00% | 99.98% | 99.99% | 0.00% | 0.02% |
| **12** | Tree | Intra-Family Voting Ensemble | **99.99%** | 100.00% | 99.98% | 99.99% | 0.00% | 0.02% |
| **13** | Probabilistic & Bayes | Quadratic Discriminant (SafeQDA) | **99.99%** | 100.00% | 99.98% | 99.99% | 0.00% | 0.02% |
| **14** | Boosting | CatBoost | **99.98%** | 100.00% | 99.94% | 99.97% | 0.00% | 0.06% |
| **15** | Boosting | LightGBM | **99.98%** | 99.98% | 99.96% | 99.97% | 0.01% | 0.04% |
| **16** | Probabilistic & Bayes | Intra-Family Voting Ensemble | **99.98%** | 99.98% | 99.98% | 99.98% | 0.01% | 0.02% |
| **17** | Probabilistic & Bayes | Gaussian Naive Bayes | **99.98%** | 99.94% | 100.00% | 99.97% | 0.04% | 0.00% |
| **18** | Probabilistic & Bayes | Bernoulli Naive Bayes | **99.98%** | 99.96% | 99.98% | 99.97% | 0.03% | 0.02% |
| **19** | Tree | Decision Tree | **99.96%** | 99.98% | 99.92% | 99.95% | 0.01% | 0.08% |

---

### 💡 Core Insights
1. **Zero False Acceptances (FAR = 0.00%)**: In high-security biometric systems, accepting an imposter is catastrophic. Feature concatenation eliminated false acceptances entirely ($FP = 0$) across 6 different classifiers.
2. **Complementary Representation**: TSBTC captures high-frequency pixel intensity distribution patterns across color bands, while VGG16 captures deep semantic facial structure and illumination gradients. Concatenating them eliminates the failure modes of each individual modality.
3. **Consistent Metric Superiority**: Precision, Recall, and F1 scores all reached $99.99\% - 100.00\%$ across 10 independent test folds.

---

## 🚀 4. Commands to Execute the Project

### Prerequisites
Make sure your terminal is inside the project directory:
```bash
cd "c:\Users\Asus\OneDrive\Desktop\BTECH FINAL YEAR_Project\Face-Spoofing-Detection\Face-Spoofing-Detection"
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

### 1️⃣ Run the ML Baseline (TSBTC Reproduction)
Runs 10-fold cross validation across all 11 $n\_parts$ variations ($n \in \{2, 4, 8, 16, 20, 24, 32, 40, 48, 56, 64\}$):
```bash
python main.py
```
*Outputs generated in `results/final_excel.xlsx` and `results/all_accuracy.csv`.*

*Fast smoke test option:*
```bash
python main.py --max-images 600 --n-values 8,32 --folds 3
```

---

### 2️⃣ Run VGG16 Deep Feature Extraction
Extracts 1,048-D continuous representations from the fine-tuned VGG16 model:
```bash
python vgg16_continuous_features.py
```

---

### 3️⃣ Run 10-Fold CV Feature Concatenation (Fused ML + DL)
Concatenates TSBTC and VGG16 features and executes full 10-fold cross-validation:

- **Run on the optimal parameter ($n=48$)**:
  ```bash
  python fuse_tsbtc_vgg16_10fold.py --n-values 48 --folds 10
  ```

- **Run on multiple parameters (e.g. $n = 8, 32, 48$)**:
  ```bash
  python fuse_tsbtc_vgg16_10fold.py --n-values 8,32,48 --folds 10
  ```

- **Run fast verification test (600 images, 3 folds)**:
  ```bash
  python fuse_tsbtc_vgg16_10fold.py --max-images 600 --n-values 8 --folds 3
  ```

---

## 📁 5. Project Directory & Artifact Map

```
Face-Spoofing-Detection/
│
├── main.py                          # TSBTC feature extraction & 10-fold CV baseline
├── vgg16_continuous_features.py      # VGG16 1048-D continuous feature extractor
├── fuse_tsbtc_vgg16_10fold.py       # Early feature concatenation & 10-fold CV fusion
├── split_80_20.py                   # 80:20 hold-out evaluation protocol
├── stack_only.py                    # Fast stacking ensemble re-computation
├── rq_summary.py                    # Research question summary generator
├── requirements.txt                 # Python library dependencies
├── archive.zip                      # Original NUAA raw image dataset archive
│
├── raw/                             # Extracted image dataset (12,614 images)
│   ├── ClientRaw/                   # Genuine human face captures (label = 1)
│   └── ImposterRaw/                 # Spoof photo attacks (label = 0)
│
└── results/                         # Generated experimental outcomes
    ├── fused_accuracy_10fold.xlsx   # Master Excel Report:
    │                                #   - Sheet 1: Final_Summary
    │                                #   - Sheet 2: TSBTC_vs_Fused_Comparison
    │                                #   - Sheet 3: All_Fused_Classifiers
    │                                #   - Sheet 4: Fused_n48
    ├── final_summary_metrics.csv    # Final comparison metrics across all stages
    ├── fused_accuracy_10fold_metrics.csv # Complete 19-classifier fused metrics
    ├── tsbtc_vs_fused_comparison.csv     # Direct gain comparison table
    ├── fused_metrics_n48.csv        # Metrics table for best configuration (n=48)
    ├── final_excel.xlsx             # Baseline ML results (TSBTC Table III)
    ├── all_accuracy.csv             # Baseline ML per-n accuracy CSV
    ├── accuracy_vs_n_parts.png      # Baseline accuracy vs n_parts curve
    └── vgg16_continuous_features_1048.npz # Cached 1048-D continuous feature matrix
```

---

## 📚 6. Mathematical Formulations

- **TSBTC Sorting (Eq. 1)**: For each color plane $C \in \{R, G, B\}$, pixel intensities are sorted in ascending order:
  $$\mathbf{S}_C = \text{sort}\big(\{P_{i,j}^C\}_{i=1,j=1}^{M,N}\big)$$
- **TSBTC Block Truncation (Eq. 2)**: $\mathbf{S}_C$ is split into $n$ parts; each part feature is its mean intensity:
  $$\mu_{C,k} = \frac{1}{|P_k|} \sum_{p \in P_k} p, \quad k = 1, \dots, n$$
- **Fused Feature Vector**:
  $$\mathbf{X}_{\text{fused}} = \left[ \frac{\mu_{R,1}}{255}, \dots, \frac{\mu_{B,n}}{255} \; \Big\| \; \mathbf{v}_{\text{VGG},1}, \dots, \mathbf{v}_{\text{VGG},1048} \right] \in [0.0, 1.0]^{1048 + 3n}$$
- **False Acceptance Rate (FAR)**: $\text{FAR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$
- **False Rejection Rate (FRR)**: $\text{FRR} = \frac{\text{FN}}{\text{FN} + \text{TP}}$

---
*Created for B.Tech Final Year Engineering Project Submission.*
