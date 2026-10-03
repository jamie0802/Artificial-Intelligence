# -*- coding: utf-8 -*-
"""
v2：所有類別 one-hot 的 NN（極端版）

類別特徵：
- batter_side: 1 or 2           -> 2 維 one-hot
- base_state: 0~3                -> 4 維 one-hot
- pitcher_id: 0~19（20 類）      -> 20 維 one-hot
- batter_type: 0~5（6 類）       -> 6 維 one-hot

連續特徵：
- balls, strikes, inning, score_diff -> 4 維

總輸入：
4 + 2 + 4 + 20 + 6 = 36 維

模型：
Input(36) -> Hidden(64, ReLU) -> Output(4, Softmax)
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


def one_hot_encode_extreme(X_raw,
                           num_pitchers=20,
                           num_btypes=6):
    balls       = X_raw[:, 0:1]
    strikes     = X_raw[:, 1:2]
    inning      = X_raw[:, 2:3]
    score_diff  = X_raw[:, 3:4]
    batter_side = X_raw[:, 4].astype(int)
    base_state  = X_raw[:, 5].astype(int)
    pitcher_id  = X_raw[:, 6].astype(int)
    batter_type = X_raw[:, 7].astype(int)

    n = X_raw.shape[0]

    # batter_side one-hot (2)
    is_left  = (batter_side == 1).astype(float).reshape(n, 1)
    is_right = (batter_side == 2).astype(float).reshape(n, 1)

    # base_state one-hot (4)
    no_run       = (base_state == 0).astype(float).reshape(n, 1)
    on1          = (base_state == 1).astype(float).reshape(n, 1)
    scoring_pos  = (base_state == 2).astype(float).reshape(n, 1)
    bases_loaded = (base_state == 3).astype(float).reshape(n, 1)

    # pitcher_id one-hot (20)
    pitcher_oh = np.eye(num_pitchers)[pitcher_id]  # (n,20)

    # batter_type one-hot (6)
    batter_oh = np.eye(num_btypes)[batter_type]    # (n,6)

    X_new = np.hstack([
        balls, strikes, inning, score_diff,          # 4
        is_left, is_right,                           # 2
        no_run, on1, scoring_pos, bases_loaded,      # 4
        pitcher_oh,                                  # 20
        batter_oh                                    # 6
    ])
    return X_new


def train_valid_split(X, y, train_ratio=0.8, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    split_idx = int(n * train_ratio)
    train_idx = idx[:split_idx]
    valid_idx = idx[split_idx:]
    return X[train_idx], y[train_idx], X[valid_idx], y[valid_idx]


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class PitchNNv2:
    def __init__(self, input_dim=36, hidden_dim=64, output_dim=4,
                 learning_rate=0.01, seed=1):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 0.1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.normal(0.0, 0.1, size=(hidden_dim, output_dim))
        self.b2 = np.zeros((1, output_dim))
        self.lr = learning_rate
        self.history_loss = []
        self.history_val_acc = []

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.h1 = np.maximum(0, self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        self.y_hat = softmax(self.z2)
        return self.y_hat

    def compute_loss(self, y_true, y_hat):
        eps = 1e-8
        y_true = y_true.reshape(-1)
        y_hat_clipped = np.clip(y_hat, eps, 1 - eps)
        n = y_true.shape[0]
        correct_probs = y_hat_clipped[np.arange(n), y_true]
        return -np.mean(np.log(correct_probs))

    def backward(self, X, y_true):
        n = X.shape[0]
        y_true = y_true.reshape(-1)
        dL_dz2 = self.y_hat.copy()
        dL_dz2[np.arange(n), y_true] -= 1
        dL_dz2 /= n

        dL_dW2 = self.h1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        dL_dh1 = dL_dz2 @ self.W2.T
        d_relu = (self.z1 > 0).astype(float)
        dL_dz1 = dL_dh1 * d_relu

        dL_dW1 = X.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        self.W2 -= self.lr * dL_dW2
        self.b2 -= self.lr * dL_db2
        self.W1 -= self.lr * dL_dW1
        self.b1 -= self.lr * dL_db1

    def train(self, X, y, X_valid=None, y_valid=None, epochs=200, print_every=50):
        for epoch in range(1, epochs + 1):
            y_hat = self.forward(X)
            loss = self.compute_loss(y, y_hat)
            self.backward(X, y)

            # 每個 epoch 都記錄 training loss
            self.history_loss.append(loss)

            # 每個 epoch 都計算並記錄 validation accuracy
            if X_valid is not None and y_valid is not None:
                val_preds = self.predict(X_valid)
                val_acc = np.mean(val_preds == y_valid)
            else:
                val_acc = 0.0
            self.history_val_acc.append(val_acc)

            # 依照 print_every 頻率輸出到終端
            if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
                preds = np.argmax(y_hat, axis=1).reshape(-1, 1)
                acc = np.mean(preds == y)
                print(f"[v2] Epoch {epoch:3d}/{epochs}, Loss={loss:.4f}, "
                      f"Train Acc={acc*100:.2f}%, "
                      f"Val Acc={val_acc*100:.2f}%")

    def predict(self, X):
        y_hat = self.forward(X)
        preds = np.argmax(y_hat, axis=1)
        return preds.reshape(-1, 1)


if __name__ == "__main__":
    print("=== v2: 類別全 one-hot（極端版） ===")
    X_raw_all, y_all = load_extreme_data_raw("pitch_train_extreme.csv")
    X_raw_test, y_test = load_extreme_data_raw("pitch_test_extreme.csv")

    X_all = one_hot_encode_extreme(X_raw_all)
    X_test = one_hot_encode_extreme(X_raw_test)

    X_train, y_train, X_valid, y_valid = train_valid_split(
        X_all, y_all, train_ratio=0.8, seed=0
    )

    print(f"Train: {X_train.shape[0]}, Valid: {X_valid.shape[0]}, Test: {X_test.shape[0]}")
    print("輸入維度:", X_train.shape[1])
    print()

    EPOCHS = 3000  # ← 想調整訓練次數只需改這裡

    model = PitchNNv2(input_dim=36, hidden_dim=64, output_dim=4,
                      learning_rate=0.01, seed=1)

    model.train(X_train, y_train,
                X_valid=X_valid, y_valid=y_valid,
                epochs=EPOCHS, print_every=50)
    print()

    print("=== Validation ===")
    yv = model.predict(X_valid)
    v_acc = np.mean(yv == y_valid)
    print(f"v2 Validation Accuracy: {v_acc*100:.2f}%")
    print()

    print("=== Test ===")
    yt = model.predict(X_test)
    t_acc = np.mean(yt == y_test)
    print(f"v2 Test Accuracy: {t_acc*100:.2f}%")
    print()

    print("說明：v2 使用 one-hot 表示 pitch_id/batter_type 等高基數類別，")
    print("請與 v1（整數編碼）比較 Validation/Test Accuracy，")
    print("觀察在「完全隨機偏好」的情況下，one-hot 對 NN 學習的影響。")

    # === 畫圖 ===
    epochs_range = range(1, EPOCHS + 1)

    # 圖一：Training Loss
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_range, model.history_loss, color="steelblue", linewidth=1.5)
    plt.title("Model A (One-hot) — Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("v2_training_loss.png", dpi=150)
    plt.show()
    print("圖一已儲存：v2_training_loss.png")

    # 圖二：Validation Accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_range, [acc * 100 for acc in model.history_val_acc],
             color="darkorange", linewidth=1.5)
    plt.title("Model A (One-hot) — Validation Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("v2_validation_accuracy.png", dpi=150)
    plt.show()
    print("圖二已儲存：v2_validation_accuracy.png")