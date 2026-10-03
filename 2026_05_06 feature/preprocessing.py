import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

df = pd.read_csv('noisy_data.csv')
water_level = df['water_level'].values

# ==============================
# EDA
# ==============================
#時間序列圖
plt.figure(figsize=(14, 4))
plt.plot(df['timestamp'], water_level, color='steelblue', lw=0.4, alpha=0.8)
plt.title('Water Level — Time Series')
plt.xlabel('Time')
plt.ylabel('water_level (m)')
plt.xticks([])
plt.tight_layout()
plt.show()

#箱型圖
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7), sharey=False)

bp_kwargs = dict(
    patch_artist=True,
    boxprops=dict(facecolor='lightblue', color='steelblue'),
    medianprops=dict(color='red', lw=2),
    flierprops=dict(marker='o', color='gray', ms=3, alpha=0.5),
)

ax1.boxplot(water_level, vert=True, showfliers=True, **bp_kwargs)
ax1.set_title('Full Range')
ax1.set_ylabel('water_level (m)')
ax1.set_xticks([1]); ax1.set_xticklabels(['water_level'])

Q1 = np.percentile(water_level, 25)
Q3 = np.percentile(water_level, 75)
IQR = Q3 - Q1

ax2.boxplot(water_level, vert=True, showfliers=True, **bp_kwargs)
ax2.set_ylim(Q1 - 3 * IQR, Q3 + 3 * IQR)
ax2.set_title('Zoomed (Normal Range)')
ax2.set_xticks([1]); ax2.set_xticklabels(['water_level'])

fig.suptitle('Water Level — Boxplot', fontweight='bold')
plt.tight_layout()
plt.show()

# ==============================
# 異常值偵測
# ==============================
# 1. IQR 方法
print("【IQR 方法】")

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

iqr_mask = (water_level < lower_bound) | (water_level > upper_bound)

print(f"Q1 = {Q1:.3f}, Q3 = {Q3:.3f}, IQR = {IQR:.3f}")
print(f"Lower Bound = {lower_bound:.3f}, Upper Bound = {upper_bound:.3f}")
print(f"偵測到異常值數量：{iqr_mask.sum()} 筆")

# IQR 異常值視覺化
plt.figure(figsize=(14, 4))
plt.plot(df['timestamp'], water_level, color='steelblue', lw=0.4, alpha=0.8, label='normal')
plt.scatter(df['timestamp'][iqr_mask], water_level[iqr_mask],
            color='red', s=15, zorder=5, label=f'outlier (n={iqr_mask.sum()})')
plt.axhline(upper_bound, color='orange', lw=1, ls='--', label=f'upper = {upper_bound:.3f}')
plt.axhline(lower_bound, color='orange', lw=1, ls='--', label=f'lower = {lower_bound:.3f}')
plt.title('IQR — Outliers Detection')
plt.ylabel('water_level (m)')
plt.xticks([])
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# ==============================
# IQR 方法 — Boxplot + 紅點離群值
# ==============================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), sharey=False)

bp_kwargs = dict(
    patch_artist=True,
    showfliers=False,           # 關掉預設 flier，自己畫紅點
    boxprops=dict(facecolor='lightblue', color='steelblue'),
    medianprops=dict(color='red', lw=2),
)

outlier_values_iqr = water_level[iqr_mask]
normal_values_iqr  = water_level[~iqr_mask]

for ax, zoom, title in [
    (ax1, False, 'Full Range'),
    (ax2, True,  'Zoomed (Normal Range)'),
]:
    ax.boxplot(water_level, vert=True, **bp_kwargs)

    # 正常點：加 jitter 避免疊在一起
    x_norm = np.random.normal(loc=1, scale=0.03, size=len(normal_values_iqr))
    ax.scatter(x_norm, normal_values_iqr,
               color='steelblue', alpha=0.2, s=3, label='Normal')

    # 離群點：紅點
    x_out = np.random.normal(loc=1, scale=0.03, size=len(outlier_values_iqr))
    ax.scatter(x_out, outlier_values_iqr,
               color='red', alpha=0.8, s=20, zorder=5,
               label=f'Outlier (n={len(outlier_values_iqr)})')

    if zoom:
        ax.set_ylim(Q1 - 3 * IQR, Q3 + 3 * IQR)

    ax.set_title(title)
    ax.set_ylabel('water_level (m)')
    ax.set_xticks([1]); ax.set_xticklabels(['water_level'])
    ax.legend(fontsize=9)

fig.suptitle('Water Level — Boxplot + IQR Outlier Detection', fontweight='bold')
plt.tight_layout()
plt.show()

# 2. Z-score 方法
print("\n【Z-score 方法】")

mean = np.mean(water_level)
std  = np.std(water_level)
z_scores = (water_level - mean) / std
threshold = 3.0

z_mask = np.abs(z_scores) > threshold

print(f"Mean = {mean:.3f}, Std = {std:.3f}")
print(f"threshold = ±{threshold}")
print(f"偵測到異常值數量：{z_mask.sum()} 筆")

# Z-score 異常值視覺化
z_upper = mean + threshold * std
z_lower = mean - threshold * std

plt.figure(figsize=(14, 4))
plt.plot(df['timestamp'], water_level, color='steelblue', lw=0.4, alpha=0.8, label='normal')
plt.scatter(df['timestamp'][z_mask], water_level[z_mask],
            color='red', s=15, zorder=5, label=f'outlier (n={z_mask.sum()})')
plt.axhline(z_upper, color='orange', lw=1, ls='--', label=f'mean+3σ = {z_upper:.3f}')
plt.axhline(z_lower, color='orange', lw=1, ls='--', label=f'mean-3σ = {z_lower:.3f}')
plt.title('Z-score — Outliers Detection')
plt.ylabel('water_level (m)')
plt.xticks([])
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# ==============================
# Z-score 方法 — Boxplot + 紅點離群值
# ==============================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), sharey=False)

outlier_values_z = water_level[z_mask]
normal_values_z  = water_level[~z_mask]

for ax, zoom, title in [
    (ax1, False, 'Full Range'),
    (ax2, True,  'Zoomed (Normal Range)'),
]:
    ax.boxplot(water_level, vert=True, **bp_kwargs)

    x_norm = np.random.normal(loc=1, scale=0.03, size=len(normal_values_z))
    ax.scatter(x_norm, normal_values_z,
               color='steelblue', alpha=0.2, s=3, label='Normal')

    x_out = np.random.normal(loc=1, scale=0.03, size=len(outlier_values_z))
    ax.scatter(x_out, outlier_values_z,
               color='red', alpha=0.8, s=20, zorder=5,
               label=f'Outlier (n={len(outlier_values_z)})')

    if zoom:
        ax.set_ylim(Q1 - 3 * IQR, Q3 + 3 * IQR)

    ax.set_title(title)
    ax.set_ylabel('water_level (m)')
    ax.set_xticks([1]); ax.set_xticklabels(['water_level'])
    ax.legend(fontsize=9)

fig.suptitle('Water Level — Boxplot + Z-score Outlier Detection', fontweight='bold')
plt.tight_layout()
plt.show()

# 3. 小結比較
print("\n【小結】")
print(f"IQR 方法找到異常值：{iqr_mask.sum()} 筆，索引：{np.where(iqr_mask)[0]}")
print(f"Z-score 方法找到異常值：{z_mask.sum()} 筆，索引：{np.where(z_mask)[0]}")

both     = iqr_mask & z_mask
only_iqr = iqr_mask & ~z_mask
only_z   = z_mask & ~iqr_mask

print(f"\n兩者都認定為異常：{both.sum()} 筆")
print(f"只有 IQR 認定：{only_iqr.sum()} 筆")
print(f"只有 Z-score 認定：{only_z.sum()} 筆")

# ==============================
# 異常值修正：線性內插
# ==============================

# 先確認要修正的異常值（這裡用兩者都認定的 both，最保守）
# 若你想改用 iqr_mask 或 z_mask 單獨的結果，換掉下面這行即可
final_mask = iqr_mask

# 把異常值的位置設為 NaN，再用線性內插填補
df_cleaned = df.copy()
df_cleaned.loc[final_mask, 'water_level'] = np.nan
df_cleaned['water_level'] = df_cleaned['water_level'].interpolate(method='linear')

print(f"\n【異常值修正】")
print(f"修正筆數：{final_mask.sum()} 筆")
print(f"修正後 water_level — mean: {df_cleaned['water_level'].mean():.3f}, std: {df_cleaned['water_level'].std():.3f}")

# 輸出清洗後的資料
df_cleaned.to_csv('noisy_data_cleaned.csv', index=False)
print("已輸出 noisy_data_cleaned.csv")

# ==============================
# 清洗前後比較圖
# ==============================

water_level_cleaned = df_cleaned['water_level'].values

# 時間序列比較
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

ax1.plot(df['timestamp'], water_level, color='steelblue', lw=0.4, alpha=0.8)
ax1.set_title('Before Cleaning')
ax1.set_ylabel('water_level (m)')
ax1.tick_params(axis='x', labelbottom=False)

ax2.plot(df['timestamp'], water_level_cleaned, color='steelblue', lw=0.4, alpha=0.8)
ax2.set_title('After Cleaning')
ax2.set_ylabel('water_level (m)')
ax2.tick_params(axis='x', labelbottom=False)

fig.suptitle('Water Level — Time Series Before vs After', fontweight='bold')
plt.tight_layout()
plt.show()

# 箱型圖比較
fig, axes = plt.subplots(1, 2, figsize=(10, 7))

for ax, data, title in [
    (axes[0], water_level,         'Before Cleaning'),
    (axes[1], water_level_cleaned, 'After Cleaning'),
]:
    ax.boxplot(data, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='steelblue'),
               medianprops=dict(color='red', lw=2),
               flierprops=dict(marker='o', color='gray', ms=3, alpha=0.5))
    ax.set_title(title)
    ax.set_ylabel('water_level (m)')
    ax.set_xticks([1]); ax.set_xticklabels(['water_level'])

fig.suptitle('Water Level — Boxplot Before vs After', fontweight='bold')
plt.tight_layout()
plt.show()

# ==============================
# 與 clean_data.csv 比對
# ==============================

df_clean = pd.read_csv('clean_data.csv')
wl_clean   = df_clean['water_level'].values
wl_noisy   = df['water_level'].values
wl_cleaned = df_cleaned['water_level'].values

# 數值指標
mae_before  = np.mean(np.abs(wl_noisy   - wl_clean))
mae_after   = np.mean(np.abs(wl_cleaned - wl_clean))
rmse_before = np.sqrt(np.mean((wl_noisy   - wl_clean) ** 2))
rmse_after  = np.sqrt(np.mean((wl_cleaned - wl_clean) ** 2))

print("【與 clean_data 比對】")
print(f"清洗前  MAE: {mae_before:.4f}  RMSE: {rmse_before:.4f}")
print(f"清洗後  MAE: {mae_after:.4f}  RMSE: {rmse_after:.4f}")

# 時間序列三線比較
plt.figure(figsize=(14, 4))
plt.plot(df['timestamp'], wl_clean,   color='green',    lw=0.8, alpha=0.9, label='clean_data')
plt.plot(df['timestamp'], wl_noisy,   color='steelblue', lw=0.4, alpha=0.5, label='noisy (original)')
plt.plot(df['timestamp'], wl_cleaned, color='red',       lw=0.4, alpha=0.7, label='noisy (cleaned)')
plt.title('Water Level — clean vs noisy vs cleaned')
plt.xlabel('Time')
plt.ylabel('water_level (m)')
plt.xticks([])
plt.legend()
plt.tight_layout()
plt.show()

# 只看異常點附近的差異（residual）
residual_before = wl_noisy   - wl_clean
residual_after  = wl_cleaned - wl_clean

plt.figure(figsize=(14, 4))
plt.plot(df['timestamp'], residual_before, color='steelblue', lw=0.4, alpha=0.6, label='Before cleaning')
plt.plot(df['timestamp'], residual_after,  color='red',       lw=0.4, alpha=0.8, label='After cleaning')
plt.axhline(0, color='black', lw=0.8, ls='--')
plt.title('Residual (noisy - clean) Before vs After')
plt.xlabel('Time')
plt.ylabel('Residual (m)')
plt.xticks([])
plt.legend()
plt.tight_layout()
plt.show()

# ==============================
# 迭代清洗（Z-score + 線性內插，每輪與 clean_data 比較 MAE / RMSE）
# ==============================

df_clean = pd.read_csv('clean_data.csv')
wl_clean = df_clean['water_level'].values

water_level_iter = df['water_level'].copy().reset_index(drop=True)
threshold = 3.0

print("=" * 55)
print(f"{'輪次':<8} {'異常數':>6} {'MAE':>10} {'RMSE':>10}")
print("=" * 55)

# 第 0 輪（原始）
mae0  = np.mean(np.abs(water_level_iter.values - wl_clean))
rmse0 = np.sqrt(np.mean((water_level_iter.values - wl_clean) ** 2))
print(f"{'原始':<8} {'—':>6} {mae0:>10.4f}  {rmse0:>10.4f}")

for i in range(10):
    mean     = water_level_iter.mean()
    std      = water_level_iter.std()
    z_scores = (water_level_iter - mean) / std
    z_mask   = np.abs(z_scores) > threshold
    n_outliers = z_mask.sum()

    if n_outliers == 0:
        print("-" * 55)
        print(f"Round {i+1}：無異常值，已收斂，停止迭代。")
        break

    water_level_iter[z_mask] = np.nan
    water_level_iter = water_level_iter.interpolate(method='linear')

    mae  = np.mean(np.abs(water_level_iter.values - wl_clean))
    rmse = np.sqrt(np.mean((water_level_iter.values - wl_clean) ** 2))
    print(f"Round {i+1:<3}  {n_outliers:>6}    {mae:>10.4f}  {rmse:>10.4f}")

else:
    print("-" * 55)
    print("達到最大迭代次數（10 輪），強制停止。")

print("=" * 55)

df_cleaned_iter = df.copy()
df_cleaned_iter['water_level'] = water_level_iter.values
df_cleaned_iter.to_csv('noisy_data_cleaned_z_score.csv', index=False)
print("已輸出 noisy_data_cleaned_z_score.csv")

# ==============================
# 最終結果與 clean_data.csv 正式比對
# ==============================

wl_final = df_cleaned_iter['water_level'].values

mae_final  = np.mean(np.abs(wl_final - wl_clean))
rmse_final = np.sqrt(np.mean((wl_final - wl_clean) ** 2))

print("\n【最終結果與 clean_data 比對】")
print(f"原始   MAE: {mae0:.4f}  RMSE: {rmse0:.4f}")
print(f"清洗後 MAE: {mae_final:.4f}  RMSE: {rmse_final:.4f}")
print(f"MAE 改善：{mae0 - mae_final:.4f}  RMSE 改善：{rmse0 - rmse_final:.4f}")