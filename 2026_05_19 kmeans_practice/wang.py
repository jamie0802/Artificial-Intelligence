import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
from scipy.optimize import linear_sum_assignment

# ── 資料預處理 ───────────────────────────────────────────────────────────────
df = pd.read_csv('iris.csv')
X = df[['petal_length', 'petal_width']].values
y_true = df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2}).values  # 僅用於事後評估，不餵給模型

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 模型訓練（完全不使用真實標籤）──────────────────────────────────────────
kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

spectral = SpectralClustering(n_clusters=3, affinity='rbf', gamma=1.0,
                               n_init=20, random_state=42)
spectral_labels = spectral.fit_predict(X_scaled)

# OneClassSVM（SVDD 風格，RBF 核）
ocsvm = OneClassSVM(kernel='rbf', gamma=0.5, nu=0.05)
ocsvm.fit(X_scaled)
support_vectors = ocsvm.support_vectors_

# ── 匈牙利演算法標籤對齊 ─────────────────────────────────────────────────────
def align_labels(y_pred, y_true, n_classes=3):
    cost_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for pred, true in zip(y_pred, y_true):
        cost_matrix[pred][true] += 1
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    aligned = np.array([mapping[p] for p in y_pred])
    return aligned, mapping

kmeans_aligned,   km_map = align_labels(kmeans_labels,  y_true)
spectral_aligned, sp_map = align_labels(spectral_labels, y_true)

km_f1 = f1_score(y_true, kmeans_aligned,  average='macro')
sp_f1 = f1_score(y_true, spectral_aligned, average='macro')

print(f"K-Means  Macro F1: {km_f1:.4f}")
print(f"Spectral Macro F1: {sp_f1:.4f}")

# ── 顏色設定 ─────────────────────────────────────────────────────────────────
COLORS    = ['#1f77b4', '#bcbd22', '#9467bd']
CENTROIDS = kmeans.cluster_centers_

# ── 繪圖 ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ════════════════════════════════════════════
# 左圖：K-Means + 橢圓虛線邊界 + 紅色 ✕ 質心
# ════════════════════════════════════════════
ax = axes[0]
ax.set_facecolor('white')

for k in range(3):
    mask = kmeans_labels == k
    ax.scatter(X_scaled[mask, 0], X_scaled[mask, 1],
               c=COLORS[k], s=40, zorder=3)

for k in range(3):
    pts  = X_scaled[kmeans_labels == k]
    mean = pts.mean(axis=0)
    cov  = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle  = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * 2.2 * np.sqrt(vals)
    ax.add_patch(Ellipse(xy=mean, width=width, height=height, angle=angle,
                         edgecolor='navy', facecolor=COLORS[k],
                         linewidth=1.8, linestyle='--', alpha=0.18, zorder=2))
    ax.add_patch(Ellipse(xy=mean, width=width, height=height, angle=angle,
                         edgecolor='navy', facecolor='none',
                         linewidth=1.8, linestyle='--', zorder=2))

ax.scatter(CENTROIDS[:, 0], CENTROIDS[:, 1],
           c='red', marker='x', s=180, linewidths=3, zorder=5)

ax.set_title('K-Means Clustering (Circular/Ellipse Boundaries)', fontsize=12)
ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.0, 2.3)
ax.legend(handles=[plt.scatter([], [], c='red', marker='x', s=100,
                               linewidths=2, label='Centroids')],
          fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)

# ════════════════════════════════════════════
# 右圖：OneClassSVM + 封閉輪廓 + Support Vectors
# ════════════════════════════════════════════
ax = axes[1]
ax.set_facecolor('white')

h = 0.02
x_min, x_max = X_scaled[:, 0].min() - .4, X_scaled[:, 0].max() + .4
y_min, y_max = X_scaled[:, 1].min() - .4, X_scaled[:, 1].max() + .4
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = ocsvm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# 填色（decision function > 0 的區域 = 正常點區域）
ax.contourf(xx, yy, Z, levels=[0, Z.max()],
            colors=['#c8e6c9'], alpha=0.6, zorder=1)
ax.contour(xx, yy, Z, levels=[0],
           colors=['#2e7d32'], linewidths=2.0, zorder=2)

# 各群資料點（沿用 K-Means 分群結果著色，方便對比）
for k in range(3):
    mask = kmeans_labels == k
    ax.scatter(X_scaled[mask, 0], X_scaled[mask, 1],
               c=COLORS[k], s=40, zorder=3)

# Support Vectors（紅圈標示）
ax.scatter(support_vectors[:, 0], support_vectors[:, 1],
           s=120, facecolors='none', edgecolors='red',
           linewidths=1.8, zorder=5, label='Support Vectors')

ax.set_title('SVC/SVDD Geometric Boundary (Kernel-Based)', fontsize=12)
ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.0, 2.3)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)

plt.suptitle(f'Iris Clustering Comparison   |   K-Means Macro F1 = {km_f1:.4f}   |   Spectral Macro F1 = {sp_f1:.4f}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('iris_style1_reference.png', dpi=180, bbox_inches='tight')
plt.show()
print("圖1（參考風格）已儲存：iris_style1_reference.png")