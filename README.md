# Interictal Clinical Signatures (ICS): Epilepsy vs Mimickers from Routine Interictal EEG

**Interpretable 13-feature biomarker + logistic regression to distinguish epilepsy from common mimickers (e.g., PNES/FND, syncope) using routine interictal EEG.**

## Dataset (multi-centre)
- **IHBAS:** N = 230 (115 epilepsy / 115 mimickers)
- **MAX:** N = 218 (109 epilepsy / 109 mimickers)
- **Total:** N = 448  
(10-second epochs; robustness tested across 20–125 Hz.) :contentReference[oaicite:0]{index=0}

## Method (high level)
- Segment EEG into 10 s epochs
- Compute **13 ICS features** (slowing, PDR, complexity, synchrony)
- **Two-stage logistic regression** for subject-level probability

## Results (subject-level AUC)
**Within-centre (5-fold CV; recording length excluded):**
- **IHBAS:** **0.723** (95% CI 0.655–0.783)
- **MAX:** **0.790** (95% CI 0.722–0.845) :contentReference[oaicite:1]{index=1}

**Cross-centre generalisation (recording length confound controlled):**
- **MAX → IHBAS:** **0.725**
- **IHBAS → MAX:** **0.725** :contentReference[oaicite:2]{index=2}

**Recording-length confound (what *not* to do):**
- If recording length is included, transfer becomes asymmetric:
  - MAX → IHBAS: 0.716
  - IHBAS → MAX: **0.360 (inverted predictions)** :contentReference[oaicite:3]{index=3}

**Fixed-duration truncation (10/20/30 min):**
- Cross-centre AUC stays ~**0.70–0.72** in both directions. :contentReference[oaicite:4]{index=4}

## Citation
To be updated.
