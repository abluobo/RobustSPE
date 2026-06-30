"""PyTorch linear regressor and training loop."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LinearRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


def train_zspace_regressor(
    x_train: np.ndarray,
    z_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    device: torch.device | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Train L1 linear regressor in z-space; return z predictions on x_eval."""
    device = device or get_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    idx = np.random.permutation(len(x_train))
    x_tr = torch.tensor(x_train[idx], dtype=torch.float32, device=device)
    y_tr = torch.tensor(z_train[idx], dtype=torch.float32, device=device)
    x_ev = torch.tensor(x_eval, dtype=torch.float32, device=device)

    model = LinearRegressor(x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=10, factor=0.3, min_lr=1e-6
    )
    loss_fn = nn.L1Loss()

    best_loss, best_state = float("inf"), None
    n = len(x_tr)
    for _ in range(EPOCHS):
        epoch_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            xb = x_tr[start : start + BATCH_SIZE]
            yb = y_tr[start : start + BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= n
        sched.step(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(x_ev).cpu().numpy()
