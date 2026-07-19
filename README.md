# Medical X-Ray Anomaly Detection

CS 435 (Applied Deep Learning) - Project 3. Investigates whether GAN-based synthetic data augmentation improves CNN classification of rare disease classes in the NIH Chest X-Ray14 dataset, which is heavily imbalanced toward "No Finding".

Five classes are used: `Pneumonia`, `Effusion`, `Cardiomegaly`, `Atelectasis`, `No Finding`.

## Approach

1. **Phase 1 — Dataset Setup & Preprocessing**: Load the NIH sample metadata, one-hot encode the 5 target labels, confirm class imbalance, resize/normalize images, and build train/val/test splits.
2. **Phase 2 — CNN Baseline**: Train a CNN on the real, unaugmented data as the control experiment.
3. **Phase 3 — Conditional GAN**: Train a GAN (Dense -> Conv2DTranspose generator, Conv2D discriminator) to synthesize 64×64 X-ray images for the minority classes.
4. **Phase 4 — CNN on Augmented Data**: Retrain the identical CNN architecture from Phase 2, this time on real data + GAN-generated synthetic images, for a fair comparison.
5. **Phase 5 — Final Comparison & Analysis**: Compare Phase 2 vs. Phase 4 metrics (AUC-ROC, F1, precision, recall) per class, with grouped bar charts and a delta heatmap showing where augmentation helped or hurt.

## Notebooks

| Notebook | Description |
|---|---|
| [`Phase_1.ipynb`](Phase_1.ipynb) | Dataset setup & preprocessing pipeline |
| [`Phase_2.ipynb`](Phase_2.ipynb) | CNN baseline (no GAN augmentation) |
| [`Phase_3.ipynb`](Phase_3.ipynb) | Conditional GAN for minority-class synthesis |
| [`Phase_4.ipynb`](Phase_4.ipynb) | CNN retrained on GAN-augmented data |
| [`Phase_5.ipynb`](Phase_5.ipynb) | Final comparison, analysis & report figures |

## Setup

Requires Python 3.9–3.11 (per TensorFlow's supported version range for `tensorflow>=2.12`).

```bash
pip install -r requirements.txt
```

Data: [NIH Chest X-Ray14 sample dataset](https://www.kaggle.com/datasets/nih-chest-xrays/sample) (download via Kaggle CLI; not included in this repo).

## Stack

TensorFlow, NumPy, pandas, scikit-learn, OpenCV, matplotlib, seaborn.
