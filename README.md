# Interpretable Multi-Centre Interictal EEG Biomarker for Epilepsy

**A machine learning framework for distinguishing epilepsy from mimickers using routine interictal EEG.**

##  Overview

This project implements an **Interictal Clinical Signature (ICS)**: a 13-dimensional biomarker designed to distinguish epilepsy from clinical mimickers (e.g., PNES, syncope). 

Unlike prior work often limited to single-centre studies with small sample sizes, this study validates the biomarker across two tertiary centres, addressing the critical challenge of acquisition differences and cross-centre generalization.

> **Significance:** The ICS framework provides an accessible, interpretable decision-support tool for epilepsy diagnostics. Our findings on asymmetric cross-centre behavior highlight the need for centre-specific calibration or domain adaptation in multi-site deployment.

##  Dataset

The study utilizes routine interictal EEG data from two tertiary care centres, encompassing diverse epilepsy subtypes and mimicker categories.

| Centre | Sample Size ($N$) | Cohort Composition |
| :--- | :--- | :--- |
| **IHBAS** | $230$ | **Epilepsy:** Generalised, focal, focal-to-bilateral, combined, syndrome-specific.<br>**Mimickers:** Psychogenic/functional, cardiac/syncope, cerebrovascular/metabolic, sleep/autonomic/movement disorders. |
| **MAX** | $218$ | Similar diversity in subtypes and mimickers. |

##  Methodology

### Preprocessing & Segmentation
* Continuous EEG recordings were segmented into **10-second epochs**.
* Robustness tested across sampling frequencies of 20–125 Hz.

### Feature Extraction: The ICS
We computed a **13-dimensional Interictal Clinical Signature (ICS)** encoding clinically grounded features:
1.  **Spectral Slowing:** Analysis of lower frequency bands.
2.  **Posterior Dominant Rhythm (PDR):** Alpha band characteristics.
3.  **Complexity:** Information content and signal entropy.
4.  **Network Synchrony:** Functional connectivity measures.

### Classification Framework
A **two-stage logistic regression** framework was used:
1.  **Stage 1:** Aggregates epoch-level features.
2.  **Stage 2:** Generates subject-level predictions.
3.  **Evaluation:** Area Under the Curve (AUC), Calibration plots, and Decision Curve Analysis.

##  Results

### Within-Centre Performance
The biomarker demonstrated robust classification performance within each centre.

| Centre | Subject-Level AUC | 95% Confidence Interval |
| :--- | :--- | :--- |
| **IHBAS** | **0.760** | 0.698 -- 0.819 |
| **MAX** | **0.790** | 0.728 -- 0.848 |

*Note: Feature-level effect sizes were counterintuitively higher in IHBAS (mean $|d| = 0.162$) compared to MAX (mean $|d| = 0.143$), suggesting performance differences stem from recording duration and calibration rather than intrinsic feature separability.*

### Cross-Centre Generalisation
Our experiments revealed a striking **asymmetric generalisation**:

* **MAX $\to$ IHBAS:** Models trained on MAX transferred reasonably well to IHBAS (AUC = **0.716**).
* **IHBAS $\to$ MAX:** Models trained on IHBAS failed to generalize, showing inverted predictions on MAX (AUC = **0.360**).

This suggests the existence of fundamentally different, centre-specific signatures requiring domain adaptation.

## 📝 Citation

To be updated...
