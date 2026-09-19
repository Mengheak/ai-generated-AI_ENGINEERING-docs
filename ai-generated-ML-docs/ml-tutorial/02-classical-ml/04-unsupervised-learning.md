# 04 · Unsupervised Learning

> **Goal:** Find structure without labels: clusters, compressed representations, anomalies.

## Clustering
| Algorithm | Idea | Pick when |
|---|---|---|
| K-Means | K centroids, assign to nearest | Round clusters, need speed; choose K via elbow/silhouette |
| DBSCAN / HDBSCAN | Dense regions = clusters, rest = noise | Arbitrary shapes, outliers present |
| Hierarchical | Merge closest clusters (dendrogram) | Small data, want a hierarchy |
| Gaussian Mixture | Soft probabilistic clusters | Overlapping clusters |

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

Xs = StandardScaler().fit_transform(X)   # ALWAYS scale for distance-based methods
for k in range(2, 9):
    labels = KMeans(n_clusters=k, n_init="auto", random_state=0).fit_predict(Xs)
    print(k, round(silhouette_score(Xs, labels), 3))
```

## Dimensionality Reduction
| Method | Use |
|---|---|
| PCA | Linear compression, speed-up, denoise; keep ~95% variance |
| t-SNE / UMAP | 2D visualization of high-dim data / embeddings (not for features) |
| Autoencoders | Non-linear compression (deep learning) |

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95).fit(Xs)
print(pca.n_components_, pca.explained_variance_ratio_.sum())
```

## Anomaly Detection
- **Isolation Forest** — fast, general-purpose.
- **Local Outlier Factor** — density-based.
- **Autoencoder reconstruction error** — for complex data.

```python
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.01, random_state=0).fit(Xs)
is_outlier = iso.predict(Xs) == -1
```

## Real Use Cases
Customer segmentation · fraud/anomaly detection · topic grouping · embedding visualization · recommender pre-processing.

## Exercises
1. Segment a mall-customers dataset with K-Means; describe each segment in business terms.
2. Visualize MNIST with PCA vs UMAP. Which separates digits better?

---
Next → [Neural Networks with PyTorch](../03-deep-learning/01-neural-networks-pytorch.md)
