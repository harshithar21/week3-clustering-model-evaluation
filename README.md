# Week 3: Unsupervised Learning & Model Evaluation

Clustering (K-Means, Hierarchical, DBSCAN), dimensionality reduction (PCA) and a full model evaluation report (cross-validation, confusion matrix, precision, recall, F1, ROC-AUC, PR curve, hyperparameter tuning, threshold selection).

## What's inside
| File | Description |
|---|---|
| `Week3_Clustering_and_Evaluation.ipynb` | Main notebook with code, outputs and explanations |
| `figures/` | Charts used in the report |
| `Week3_Report.docx` | Written report |

Both datasets (`load_wine`, `load_breast_cancer`) come with scikit-learn, so no download is needed.

## Part A: Clustering on the Wine dataset (178 wines, 13 features)
| Method | Silhouette | ARI vs true labels | NMI |
|---|---|---|---|
| K-Means (k=3) | 0.285 | 0.897 | 0.876 |
| Hierarchical (Ward) | 0.277 | 0.790 | 0.786 |

- k = 3 chosen with silhouette, Davies-Bouldin and Calinski-Harabasz (elbow curve alone was not clear)
- K-Means on **unscaled** data: ARI drops to 0.371, so scaling matters
- Single and average linkage collapse into one giant cluster (ARI about 0)
- DBSCAN performed poorly in 13 dimensions (ARI at most 0.33)

## Part B: Model evaluation on the Breast Cancer dataset (569 tumours, 30 features; malignant = positive class)
| Model (test set) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.965 | 0.975 | 0.929 | 0.951 | 0.996 |
| KNN | 0.956 | 0.974 | 0.905 | 0.938 | 0.982 |
| SVM (RBF) | 0.974 | 1.000 | 0.929 | 0.963 | 0.995 |
| Random Forest | 0.965 | 1.000 | 0.905 | 0.950 | 0.994 |

- 5-fold stratified cross-validation, with scaling/PCA inside pipelines (no data leakage)
- GridSearchCV and RandomizedSearchCV tuning: helped Random Forest, left Logistic Regression and SVM unchanged, slightly hurt KNN on the test set
- Threshold analysis: lowering the threshold from 0.5 to 0.2 cut missed cancers from 3 to 1 at the cost of more false alarms
- PCA: 10 of 30 components give the same F1 as all 30 features
- The test set is small (114 tumours), so differences between models are within noise

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
jupyter notebook Week3_Clustering_and_Evaluation.ipynb
```
