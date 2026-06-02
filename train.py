import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from model import UNet
from dataset import HeatSourceDataset

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
SAVE_PATH   = 'checkpoints'
os.makedirs(SAVE_PATH, exist_ok=True)

print(f"Training on: {DEVICE}")

# ── Data ──────────────────────────────────────────────────────────────────────
train_ds = HeatSourceDataset('data/train.npz')
val_ds   = HeatSourceDataset('data/val.npz')

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Model ─────────────────────────────────────────────────────────────────────
model     = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
criterion = nn.MSELoss()

# ── Physics-Informed Loss (for publication) ───────────────────────────────────
def pde_residual_loss(T_pred_Q, T_input, dx=1.0/64, k=50.0):
    """
    Penalizes violation of ∇²T = -Q/k
    T_input: normalized temperature (we un-normalize conceptually)
    T_pred_Q: predicted heat source map (normalized)
    """
    # Laplacian of T using finite differences
    lap_T = (
        torch.roll(T_input, 1, -1) + torch.roll(T_input, -1, -1) +
        torch.roll(T_input, 1, -2) + torch.roll(T_input, -1, -2) -
        4 * T_input
    ) / (dx ** 2)
    residual = lap_T + T_pred_Q / k
    return torch.mean(residual ** 2)

# ── Training Loop ─────────────────────────────────────────────────────────────
best_val_loss = float('inf')
train_losses, val_losses = [], []

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0.0
    for T_batch, Q_batch in train_loader:
        T_batch, Q_batch = T_batch.to(DEVICE), Q_batch.to(DEVICE)

        optimizer.zero_grad()
        Q_pred = model(T_batch)

        mse  = criterion(Q_pred, Q_batch)
        pde  = pde_residual_loss(Q_pred, T_batch)
        loss = mse + 0.01 * pde          # λ=0.01 balances data vs physics loss

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for T_batch, Q_batch in val_loader:
            T_batch, Q_batch = T_batch.to(DEVICE), Q_batch.to(DEVICE)
            Q_pred = model(T_batch)
            val_loss += criterion(Q_pred, Q_batch).item()
    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'{SAVE_PATH}/best_model.pth')

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Best: {best_val_loss:.6f}")

# Save loss history
np.save(f'{SAVE_PATH}/train_losses.npy', np.array(train_losses))
np.save(f'{SAVE_PATH}/val_losses.npy',   np.array(val_losses))
print(f"\nDone. Best val loss: {best_val_loss:.6f}")
print(f"Model saved to {SAVE_PATH}/best_model.pth")