# -*- coding: utf-8 -*-
"""
v3：高基數類別改用 Embedding 的 NN（極端版）

類別特徵：
- batter_side: 1 or 2           -> 2 維 one-hot  （基數小，維持 one-hot）
- base_state: 0~3                -> 4 維 one-hot  （基數小，維持 one-hot）
- pitcher_id: 0~19（20 類）      -> 4 維 embedding（高基數，改用 embedding）
- batter_type: 0~5（6 類）       -> 2 維 embedding（中基數，改用 embedding）

連續特徵：
- balls, strikes, inning, score_diff -> 4 維

總輸入（送進 MLP 前）：
4 + 2 + 4 + 4 + 2 = 16 維

模型（與 v2 對齊）：
Input(16) -> Hidden(64, ReLU) -> Output(4, Softmax)
Loss / Optimizer / Epoch / learning_rate 皆與 v2 相同
"""

import numpy as np
import csv
import matplotlib.pyplot as plt


def load_extreme_data_raw(filename):
    X_list = []
    y_list = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            try:
                balls       = float(row[0])
                strikes     = float(row[1])
                inning      = float(row[2])
                score_diff  = float(row[3])
                batter_side = int(float(row[4]))  # 1 or 2
                base_state  = int(float(row[5]))  # 0~3
                pitcher_id  = int(float(row[6]))  # 0~19
                batter_type = int(float(row[7]))  # 0~5
                y_val       = int(float(row[8]))
            except (ValueError, IndexError):
                continue
            X_list.append([
                balls, strikes, inning, score_diff,
                batter_side, base_state, pitcher_id, batter_type
            ])
            y_list.append([y_val])
    X_raw = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)
    return X_raw, y


def embedding_encode(X_raw):
    """
    只處理「不需要 embedding」的部分：
    - 連續特徵：balls, strikes, inning, score_diff（4維）
    - batter_side one-hot（2維）
    - base_state  one-hot（4維）

    pitcher_id、batter_type 保留為整數索引，
    留給模型內部的 embedding 矩陣處理。

    回傳：
      X_base      : (N, 10)  連續 + 小類別 one-hot
      pitcher_ids : (N,)     整數索引，供 embedding lookup
      batter_ids  : (N,)     整數索引，供 embedding lookup
    """
    n = X_raw.shape[0]

    balls       = X_raw[:, 0:1]
    strikes     = X_raw[:, 1:2]
    inning      = X_raw[:, 2:3]
    score_diff  = X_raw[:, 3:4]
    batter_side = X_raw[:, 4].astype(int)
    base_state  = X_raw[:, 5].astype(int)
    pitcher_ids = X_raw[:, 6].astype(int)   # 保留整數
    batter_ids  = X_raw[:, 7].astype(int)   # 保留整數

    # batter_side one-hot（2維）
    is_left  = (batter_side == 1).astype(float).reshape(n, 1)
    is_right = (batter_side == 2).astype(float).reshape(n, 1)

    # base_state one-hot（4維）
    no_run       = (base_state == 0).astype(float).reshape(n, 1)
    on1          = (base_state == 1).astype(float).reshape(n, 1)
    scoring_pos  = (base_state == 2).astype(float).reshape(n, 1)
    bases_loaded = (base_state == 3).astype(float).reshape(n, 1)

    # 拼成 base 特徵：4 + 2 + 4 = 10 維
    X_base = np.hstack([
        balls, strikes, inning, score_diff,      # 4
        is_left, is_right,                        # 2
        no_run, on1, scoring_pos, bases_loaded,   # 4
    ])

    return X_base, pitcher_ids, batter_ids


def train_valid_split(X, y, train_ratio=0.8, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    split_idx = int(n * train_ratio)
    train_idx = idx[:split_idx]
    valid_idx = idx[split_idx:]
    return (X[train_idx], y[train_idx],
            X[valid_idx], y[valid_idx],
            train_idx, valid_idx)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class PitchNNv3:
    def __init__(self,
                 input_dim=16, hidden_dim=64, output_dim=4,
                 num_pitchers=20, pitcher_emb_dim=4,
                 num_btypes=6,   batter_emb_dim=2,
                 learning_rate=0.01, seed=1):
        rng = np.random.default_rng(seed)

        # Embedding 矩陣（可訓練參數）
        self.pitcher_emb = rng.normal(0.0, 0.1, size=(num_pitchers, pitcher_emb_dim))
        self.batter_emb  = rng.normal(0.0, 0.1, size=(num_btypes,   batter_emb_dim))

        # MLP 權重（與 v2 相同結構）
        self.W1 = rng.normal(0.0, 0.1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.normal(0.0, 0.1, size=(hidden_dim, output_dim))
        self.b2 = np.zeros((1, output_dim))

        self.lr = learning_rate
        self.history_loss    = []
        self.history_val_acc = []

    def forward(self, X_base, pitcher_ids, batter_ids):
        """
        X_base      : (N, 10) 連續 + 小類別 one-hot
        pitcher_ids : (N,)    整數索引
        batter_ids  : (N,)    整數索引
        """
        # Embedding lookup：整數索引 → 稠密向量
        p_emb = self.pitcher_emb[pitcher_ids]   # (N, 4)
        b_emb = self.batter_emb[batter_ids]     # (N, 2)

        # 拼接成完整輸入：10 + 4 + 2 = 16 維
        X = np.hstack([X_base, p_emb, b_emb])  # (N, 16)
        self.X_input = X                         # 存起來給 backward 用

        # 儲存索引供 backward 更新對應 embedding 列
        self.pitcher_ids = pitcher_ids
        self.batter_ids  = batter_ids

        # MLP forward（與 v2 完全相同）
        self.z1    = X @ self.W1 + self.b1
        self.h1    = np.maximum(0, self.z1)
        self.z2    = self.h1 @ self.W2 + self.b2
        self.y_hat = softmax(self.z2)
        return self.y_hat

    def compute_loss(self, y_true, y_hat):
        eps = 1e-8
        y_true = y_true.reshape(-1)
        y_hat_clipped = np.clip(y_hat, eps, 1 - eps)
        n = y_true.shape[0]
        correct_probs = y_hat_clipped[np.arange(n), y_true]
        return -np.mean(np.log(correct_probs))

    def backward(self, X_base, y_true):
        n = X_base.shape[0]
        y_true = y_true.reshape(-1)

        # === MLP backward（與 v2 相同）===
        dL_dz2 = self.y_hat.copy()
        dL_dz2[np.arange(n), y_true] -= 1
        dL_dz2 /= n

        dL_dW2 = self.h1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        dL_dh1 = dL_dz2 @ self.W2.T
        d_relu  = (self.z1 > 0).astype(float)
        dL_dz1  = dL_dh1 * d_relu

        dL_dW1 = self.X_input.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        self.W2 -= self.lr * dL_dW2
        self.b2 -= self.lr * dL_db2
        self.W1 -= self.lr * dL_dW1
        self.b1 -= self.lr * dL_db1

        # === Embedding backward ===
        # dL_dX_input 的最後 6 維對應 embedding 部分
        dL_dX = dL_dz1 @ self.W1.T           # (N, 16)
        dL_dp_emb = dL_dX[:, 10:14]          # (N, 4)  對應 pitcher embedding
        dL_db_emb = dL_dX[:, 14:16]          # (N, 2)  對應 batter  embedding

        # 只更新本 batch 實際用到的 embedding 列（用 np.add.at 累加梯度）
        np.add.at(self.pitcher_emb, self.pitcher_ids, -self.lr * dL_dp_emb)
        np.add.at(self.batter_emb,  self.batter_ids,  -self.lr * dL_db_emb)

    def train(self, X_base, y, pitcher_ids, batter_ids,
              X_base_valid=None, y_valid=None,
              pitcher_ids_valid=None, batter_ids_valid=None,
              epochs=200, print_every=50):

        for epoch in range(1, epochs + 1):
            y_hat = self.forward(X_base, pitcher_ids, batter_ids)
            loss  = self.compute_loss(y, y_hat)
            self.backward(X_base, y)

            # 記錄 training loss
            self.history_loss.append(loss)

            # 計算並記錄 validation accuracy
            if X_base_valid is not None and y_valid is not None:
                val_preds = self.predict(X_base_valid, pitcher_ids_valid, batter_ids_valid)
                val_acc   = np.mean(val_preds == y_valid)
            else:
                val_acc = 0.0
            self.history_val_acc.append(val_acc)

            # 依照 print_every 頻率輸出到終端
            if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
                preds = np.argmax(y_hat, axis=1).reshape(-1, 1)
                acc   = np.mean(preds == y)
                print(f"[v3] Epoch {epoch:3d}/{epochs}, Loss={loss:.4f}, "
                      f"Train Acc={acc*100:.2f}%, "
                      f"Val Acc={val_acc*100:.2f}%")

    def predict(self, X_base, pitcher_ids, batter_ids):
        y_hat = self.forward(X_base, pitcher_ids, batter_ids)
        preds = np.argmax(y_hat, axis=1)
        return preds.reshape(-1, 1)


if __name__ == "__main__":
    print("=== v3: 高基數類別用 Embedding（極端版）===")

    X_raw_all,  y_all  = load_extreme_data_raw("pitch_train_extreme.csv")
    X_raw_test, y_test = load_extreme_data_raw("pitch_test_extreme.csv")

    # Embedding encode：回傳 base 特徵 + 整數索引
    X_base_all,  pitcher_ids_all,  batter_ids_all  = embedding_encode(X_raw_all)
    X_base_test, pitcher_ids_test, batter_ids_test = embedding_encode(X_raw_test)

    # 切分 train / valid（seed=0 與 v2 一致，確保公平比較）
    (X_base_train, y_train,
     X_base_valid, y_valid,
     train_idx, valid_idx) = train_valid_split(X_base_all, y_all, train_ratio=0.8, seed=0)

    pitcher_ids_train = pitcher_ids_all[train_idx]
    batter_ids_train  = batter_ids_all[train_idx]
    pitcher_ids_valid = pitcher_ids_all[valid_idx]
    batter_ids_valid  = batter_ids_all[valid_idx]

    print(f"Train: {X_base_train.shape[0]}, Valid: {X_base_valid.shape[0]}, Test: {X_base_test.shape[0]}")
    print("MLP 輸入維度: 16  (10 base + 4 pitcher_emb + 2 batter_emb)")
    print()

    EPOCHS = 5000  # ← 想調整訓練次數只需改這裡

    model = PitchNNv3(
        input_dim=16, hidden_dim=64, output_dim=4,
        num_pitchers=20, pitcher_emb_dim=4,
        num_btypes=6,   batter_emb_dim=2,
        learning_rate=0.01, seed=1
    )

    model.train(
        X_base_train, y_train, pitcher_ids_train, batter_ids_train,
        X_base_valid=X_base_valid, y_valid=y_valid,
        pitcher_ids_valid=pitcher_ids_valid, batter_ids_valid=batter_ids_valid,
        epochs=EPOCHS, print_every=50
    )
    print()

    print("=== Validation ===")
    yv    = model.predict(X_base_valid, pitcher_ids_valid, batter_ids_valid)
    v_acc = np.mean(yv == y_valid)
    print(f"v3 Validation Accuracy: {v_acc*100:.2f}%")
    print()

    print("=== Test ===")
    yt    = model.predict(X_base_test, pitcher_ids_test, batter_ids_test)
    t_acc = np.mean(yt == y_test)
    print(f"v3 Test Accuracy: {t_acc*100:.2f}%")
    print()

    print("說明：v3 使用 embedding 表示 pitcher_id / batter_type 等高基數類別，")
    print("請與 v2（one-hot）比較 Validation/Test Accuracy，")
    print("觀察 embedding 在稠密表示上對 NN 學習的影響。")

    # === 畫圖 ===
    epochs_range = range(1, EPOCHS + 1)

    # 圖一：Training Loss
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_range, model.history_loss, color="steelblue", linewidth=1.5)
    plt.title("Model B (Embedding) — Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("v3_training_loss.png", dpi=150)
    plt.show()
    print("圖一已儲存：v3_training_loss.png")

    # 圖二：Validation Accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_range, [acc * 100 for acc in model.history_val_acc],
             color="darkorange", linewidth=1.5)
    plt.title("Model B (Embedding) — Validation Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("v3_validation_accuracy.png", dpi=150)
    plt.show()
    print("圖二已儲存：v3_validation_accuracy.png")