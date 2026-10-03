import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.svm import OneClassSVM
from scipy.optimize import linear_sum_assignment

df = pd.read_csv("iris.csv")
X = df.drop(columns=["sepal_length","sepal_width","species"])
y = df["species"]

#前處理
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

le = LabelEncoder()
y_le = le.fit_transform(y) #把文字轉成數字對應
class_names = le.classes_
print(class_names) #確認一下轉換後的類別對應的編號

#訓練KMeans和Spectral Clustering模型
kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10) #K告訴模型要分三群
kmeans_labels = kmeans_model.fit_predict(X_std) #先訓練KMeans模型，再直接回傳每筆資料的群集編號

spectral_model = SpectralClustering(n_clusters=3, affinity='rbf', gamma=1.0, random_state=42, n_init=20)
spectral_labels = spectral_model.fit_predict(X_std)

#標籤對齊
def align_labels(y_true, y_pred, n_classes=3):
    """
    建立混淆矩陣，用匈牙利演算法找最佳 cluster→class 映射。
    回傳對齊後的預測標籤與映射字典。
    """
    cost_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for pred, true in zip(y_pred, y_true):
        cost_matrix[pred][true] += 1                    # 統計每個 cluster 裡各類別的數量

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)  # 負號：求最大對齊數
    mapping  = {r: c for r, c in zip(row_ind, col_ind)}
    y_aligned = np.array([mapping[p] for p in y_pred])     # 套用映射，翻譯所有標籤
    return y_aligned, mapping

#實際對應
kmeans_aligned,   kmeans_mapping   = align_labels(y_le, kmeans_labels)
spectral_aligned, spectral_mapping = align_labels(y_le, spectral_labels)

#計算指標(Macro F1 score)
kmeans_f1 = f1_score(y_le, kmeans_aligned, average='macro')
spectral_f1 = f1_score(y_le, spectral_aligned, average='macro')

print(f"K-Means   Macro F1 : {kmeans_f1:.4f}")
print(f"Spectral  Macro F1 : {spectral_f1:.4f}")

#One-Class SVM
oc_svm = OneClassSVM(kernel='rbf', gamma=0.5, nu=0.05)
oc_svm.fit(X_std)

sv_all = oc_svm.support_vectors_

# 網格（標準化空間）
margin = 0.3
x_min, x_max = X_std[:, 0].min() - margin, X_std[:, 0].max() + margin
y_min, y_max = X_std[:, 1].min() - margin, X_std[:, 1].max() + margin
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
grid = np.c_[xx.ravel(), yy.ravel()]

Z = oc_svm.decision_function(grid).reshape(xx.shape)
 
#視覺化
colors_pt = ['#2471A3', '#1E8449', '#B7950B']
 
def draw_ellipse(points, ax, color, n_std=2.0):
    """
    對群集資料的共變異數矩陣做特徵值分解，畫出 n_std 個標準差範圍的橢圓。
    能反映 K-Means「以距離為中心的球形思維」。
    """
    mean = points.mean(axis=0)
    cov  = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle  = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse(
        xy=mean, width=width, height=height, angle=angle,
        facecolor=color, alpha=0.18,
        edgecolor=color, linewidth=2.0, linestyle='--'
    )
    ax.add_patch(ellipse)
 
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Iris Dataset: K-Means vs SVC/SVDD Geometric Boundary\n"
    "Features: petal length × petal width",
    fontsize=13, fontweight='bold'
)
 
#K-Means+橢圓邊界
ax = axes[0]
ax.set_facecolor('#F8F9FA')
 
for cluster_id in range(3):
    mask     = kmeans_labels == cluster_id
    class_id = kmeans_mapping[cluster_id]
    draw_ellipse(X_std[mask], ax, colors_pt[class_id])
 
for cid in range(3):
    mask = y_le == cid
    ax.scatter(X_std[mask, 0], X_std[mask, 1],
               color=colors_pt[cid], edgecolors='white',
               linewidths=0.5, s=55, zorder=3, label=class_names[cid])
 
for c in kmeans_model.cluster_centers_:
    ax.scatter(c[0], c[1], marker='X', s=220,
               color='red', edgecolors='darkred', linewidths=1.2, zorder=6)
ax.scatter([], [], marker='X', s=180, color='red',
           edgecolors='darkred', label='Centroids')
 
ax.set_title(f"K-Means Clustering (Circular/Ellipse Boundaries)\nMacro F1 = {kmeans_f1:.4f}",
             fontsize=11, fontweight='bold')
ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.25, linestyle=':')
ax.set_xlabel("Petal Length (standardized)", fontsize=10)
ax.set_ylabel("Petal Width (standardized)",  fontsize=10)
 

ax = axes[1]
ax.set_facecolor('#F8F9FA')

# 一個填色區域（內部為正，Z>0）＋一條邊界線（Z=0）
ax.contourf(xx, yy, Z, levels=[0, Z.max()], colors=['#A9DFBF'], alpha=0.35)
ax.contour(xx, yy, Z, levels=[0], colors=['#1E8449'], linewidths=2.2)

for cid in range(3):
    mask = y_le == cid
    ax.scatter(X_std[mask, 0], X_std[mask, 1],
               color=colors_pt[cid], edgecolors='white',
               linewidths=0.5, s=55, zorder=3, label=class_names[cid])

# 所有 SV 一次畫完（不分類別）
ax.scatter(sv_all[:, 0], sv_all[:, 1], s=150,
           facecolors='none', edgecolors='red', linewidths=1.8, zorder=5,
           label='Support Vectors')
 
ax.set_title(f"SVC/SVDD Geometric Boundary (Kernel-Based)\nMacro F1 = {spectral_f1:.4f}",
             fontsize=11, fontweight='bold')
ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.25, linestyle=':')
ax.set_xlabel("Petal Length (standardized)", fontsize=10)
ax.set_ylabel("Petal Width (standardized)",  fontsize=10)
 
plt.tight_layout()
plt.savefig("412411471.png", dpi=150, bbox_inches='tight')
plt.show()
print("完成！圖片已儲存為 412411471.png")