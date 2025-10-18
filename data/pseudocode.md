# Pseudocode — Evaluation Metrics for MedFusion-GP-AKI External Cohort

This pseudocode describes how the performance metrics were calculated for the external validation of the MedFusion-GP-AKI model, including AUC, AP, DCA, Calibration, and Threshold-based metrics.

```python
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, recall_score, precision_score, accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

# Load data
df = pd.read_csv("test1_rounded.csv")
y_true = df["AKI"]
y_score = df["ModelScore"]

# 1. AUC
AUC = roc_auc_score(y_true, y_score)
boot_auc = [roc_auc_score(*resample(y_true, y_score)) for _ in range(2000)]
CI_AUC = np.percentile(boot_auc, [2.5, 97.5])

# 2. AP
AP = average_precision_score(y_true, y_score)
boot_ap = [average_precision_score(*resample(y_true, y_score)) for _ in range(2000)]
CI_AP = np.percentile(boot_ap, [2.5, 97.5])

# 3. DCA
thresholds = np.linspace(0.01, 0.99, 99)
net_benefits = []
for t in thresholds:
    pred = y_score >= t
    TP = np.sum((pred == 1) & (y_true == 1))
    FP = np.sum((pred == 1) & (y_true == 0))
    N = len(y_true)
    NB = TP / N - FP / N * (t / (1 - t))
    net_benefits.append(NB)
prevalence = np.mean(y_true)
NB_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
NB_none = np.zeros_like(thresholds)
Mean_NB = np.mean(net_benefits)

# 4. Calibration Metrics
BS = brier_score_loss(y_true, y_score)
BSS = 1 - BS / np.var(y_true)
p = np.clip(y_score, 1e-6, 1 - 1e-6)
logit_p = np.log(p / (1 - p)).reshape(-1, 1)
model = LogisticRegression(fit_intercept=True).fit(logit_p, y_true)
CITL = model.intercept_[0]
Slope = model.coef_[0][0]
boot_slope = []
for _ in range(2000):
    idx = np.random.choice(len(y_true), len(y_true), replace=True)
    m = LogisticRegression(fit_intercept=True).fit(logit_p[idx], y_true[idx])
    boot_slope.append(m.coef_[0][0])
CI_Slope = np.percentile(boot_slope, [2.5, 97.5])

# 5. Threshold-based Metrics
th = 0.5
pred_bin = (y_score >= th).astype(int)
Sensitivity = recall_score(y_true, pred_bin)
TN, FP, FN, TP = confusion_matrix(y_true, pred_bin).ravel()
Specificity = TN / (TN + FP)
Precision = precision_score(y_true, pred_bin)
Accuracy = accuracy_score(y_true, pred_bin)
F1 = f1_score(y_true, pred_bin)

# Output summary
results = pd.DataFrame({
    "Metric": ["AUC", "AUC_95%_CI", "AP", "AP_95%_CI", "Mean_NB(0-1)", "BS", "BSS", "CITL", "Slope", "Slope_95%_CI", "Sensitivity", "Specificity", "Precision", "Accuracy", "F1"],
    "Value": [AUC, f"{CI_AUC[0]:.3f}-{CI_AUC[1]:.3f}", AP, f"{CI_AP[0]:.3f}-{CI_AP[1]:.3f}", Mean_NB, BS, BSS, CITL, Slope, f"{CI_Slope[0]:.3f}-{CI_Slope[1]:.3f}", Sensitivity, Specificity, Precision, Accuracy, F1]
})
results.to_csv("metrics_summary.csv", index=False)
print(results)
