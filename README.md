# Predictive Maintenance — Transformer vs LSTM
### Deep Learning Project

A complete pipeline for predicting industrial machine failures from multivariate time series sensor data.

---

## Project structure

```
predictive_maintenance/
│
├── data/
│   └── generate_dataset.py   ← synthetic sensor data with 3 failure modes
│
├── models/
│   └── transformer.py        ← PatchTST classifier + LSTM baseline
│
├── utils/
│   └── dataset.py            ← PyTorch Dataset, normalization, oversampling
│
├── train.py                  ← main training + evaluation script
└── README.md
```

---

## Quickstart

```bash
# 1. install dependencies
pip install torch numpy pandas scikit-learn

# 2. generate dataset + train both models
python train.py

# 3. train only the transformer
python train.py --model transformer

# 4. regenerate fresh data, train 20 epochs
python train.py --regen --epochs 20
```

---

## Dataset

**4 sensor channels** sampled at 1 Hz:

| Channel     | Baseline | Unit  |
|-------------|----------|-------|
| Temperature | 75 °C    | °C    |
| Vibration   | 1.2      | mm/s  |
| Pressure    | 5.0      | bar   |
| RPM         | 1450     | RPM   |

**3 injected failure modes:**

| Mode          | Signature                                      |
|---------------|------------------------------------------------|
| Overheat      | Temperature ramp → spike over ~200 steps       |
| Bearing wear  | Vibration rises + RPM drops over ~300 steps    |
| Pressure drop | Pressure falls sharply over ~120 steps         |

**Label:** `1` if a failure event occurs within the next **30 timesteps**, else `0`.  
Sequences: 50,000 timesteps → ~49,940 windows of length 60.  
Class split: ~3–5% failures (realistic for industrial settings).

---

## Model: PatchTST Classifier

```
Input (B, T=60, C=4)
  │
  ▼  per channel (channel-independent)
PatchEmbedding(patch_len=12)  →  5 patches per channel
  │
  ▼
Transformer Encoder (3 layers, 4 heads, d_model=64)
  │
  ▼
Mean-pool patches  →  (B, 64) per channel
  │
  ▼
Concatenate all channels  →  (B, 256)
  │
  ▼
MLP Head  →  (B, 2)  binary logit
```

**Why patches?**  
Point-wise tokenisation (one token per timestep) loses local temporal structure and creates long sequences that strain attention. Patches group neighbouring timesteps — similar to how ViT treats image pixels — giving each token richer context and reducing sequence length by `patch_len`×.

---

## Baseline: Bidirectional LSTM

```
Input (B, T=60, C=4)
  │
  ▼
BiLSTM (2 layers, hidden=64)
  │
  ▼
Last hidden state  →  (B, 128)
  │
  ▼
MLP Head  →  (B, 2)
```

---

## Key design decisions

**Class imbalance** — failures are rare (~3–5%).  
→ `WeightedRandomSampler` oversamples failures during training.  
→ `CrossEntropyLoss` with positive class weight as additional correction.

**Normalisation** — z-score per channel, statistics fit on train set only, applied to val/test.

**Early stopping** — monitors validation F1 (not loss), since F1 reflects minority-class performance.

**Metric focus** — F1 and ROC-AUC are the primary metrics. Accuracy is misleading on imbalanced data (a model predicting "normal" always gets ~97% accuracy).

---

## Expected results (CPU, 40 epochs)

| Model    | Val F1  | Test ROC-AUC |
|----------|---------|--------------|
| PatchTST | ~0.75–0.85 | ~0.90–0.95 |
| LSTM     | ~0.68–0.78 | ~0.87–0.93 |

Results vary by random seed and failure rate. The Transformer typically edges out LSTM on longer failure precursor patterns (bearing wear), while LSTM can match it on sharp events (pressure drop).

---

## Experiment ideas for your report

1. **Ablation on patch size** — try patch_len ∈ {6, 12, 20, 30}
2. **Ablation on sequence length** — does a 120-step window help?
3. **Attention visualisation** — extract attention weights from the encoder to see which patches the model focuses on before a failure
4. **Failure-mode breakdown** — evaluate separately on each failure type
5. **Label horizon** — change FAILURE_HORIZON from 30 to 10 or 60 steps
