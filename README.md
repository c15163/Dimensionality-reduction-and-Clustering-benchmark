# Dimensionality Reduction and Clustering Benchmark

This repository contains a full machine learning benchmark evaluating multiple dimensionality reduction methods and clustering algorithms, followed by neural network classification experiments.  
All analyses and experimental results are reproduced from the accompanying report.

The project includes:

- Unsupervised clustering (K-Means, EM/GMM)
- Dimensionality reduction (PCA, ICA, Random Projection, LLE)
- Clustering after dimensionality reduction
- Neural network classification before & after dimensionality reduction
- Neural network classification after clustering feature augmentation

This repository includes both:
1. Full analysis report (PDF)
2. A unified experiment script (`ML_dimension_reduction_and_clustering.py`) that reproduces all figures and results

---

## Project Structure

```
project_root/
│
├── ML_dimension_reduction_and_clustering.py                      # Unified experiment code (all parts)
├── dimension_reduction_and_clustering_report.pdf          # Full project analysis and figures
└── README.md                   # Documentation (this file)
```

---

## Datasets Used

This project uses the following two datasets.

### 1. Breast Cancer Wisconsin Dataset  
Binary classification, 30 features.

Download:  
https://www.kaggle.com/datasets/anacoder1/wisc-bc-data

### 2. WiFi Localization Dataset  
Multiclass indoor localization (4 classes), 7 WiFi access points.

Download:  
https://archive.ics.uci.edu/dataset/422/wireless+indoor+localization

Place both CSV files in:

```
project_root/
└── data/
    ├── wisc_bc_data.csv
    └── wifi_localization.csv
```

---

## Summary of Experiments

### Part 1 — Clustering (No Dimensionality Reduction)
- K-Means: inertia curves, silhouette analysis
- EM (Gaussian Mixture): BIC and silhouette analysis
- Optimal cluster selection
- Figures 1–8 in the report

### Part 2 — Dimensionality Reduction (PCA, ICA, RP, LLE)
- PCA explained variance
- ICA kurtosis analysis
- Random Projection reconstruction error (100 trials)
- LLE reconstruction error with multiple neighbors
- Figures 9–16 in the report

### Part 3 — Clustering After Dimensionality Reduction
- K-Means after PCA/ICA/RP/LLE
- EM after PCA/ICA/RP/LLE
- Silhouette evaluation comparisons
- Figures 17–25 in the report

### Part 4 — Neural Network After Dimensionality Reduction
- Performance of NN on PCA/ICA/RP/LLE-transformed data
- Training curves & loss curves
- Accuracy/time comparison
- Figures 26–35 in the report

### Part 5 — Neural Network After Clustering Feature Augmentation
- Appending cluster labels as new features
- NN performance with K-Means and EM cluster features
- Figures 36–41 in the report

---

## How to Run

Install required packages:

```bash
pip install numpy pandas scikit-learn mlrose-hiive matplotlib
```

Then run:

```bash
python ML_dimension_reduction_and_clustering.py
```

All figures will be saved automatically in the working directory.

---

## Key Conclusions

- PCA and ICA generally improved NN classification more than RP and LLE.  
- K-Means and EM behaved differently depending on the dimensionality reduction technique.
- Random Projection showed higher reconstruction error variance.
- NN performance was not improved by cluster labels (K-Means/EM), consistent with report findings.
- Full discussion and figure references appear in the PDF report.

---

## Reference  
See the project report (`hji61_analysis.pdf`) for detailed analysis, figures, and conclusions.
