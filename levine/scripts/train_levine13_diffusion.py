from pathlib import Path
import argparse
import numpy as np
import torch

from src.models.diffusion import DiffusionModel
from src.utils.device import resolve_device


def make_dataset(mode="top10_labeled", seed=42):
    data = np.load("data/levine13/levine13_processed.npz", allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    label_names = data["label_names"]

    label_names_lower = np.array([str(s).lower() for s in label_names])
    unassigned_ids = np.where(
        np.char.find(label_names_lower.astype(str), "unassigned") >= 0
    )[0]

    if len(unassigned_ids) > 0:
        unassigned_id = int(unassigned_ids[0])
        labeled_mask = y != unassigned_id
    else:
        labeled_mask = np.ones(len(y), dtype=bool)

    if mode == "all":
        mask = np.ones(len(y), dtype=bool)

    elif mode == "labeled":
        mask = labeled_mask

    elif mode == "top10_labeled":
        counts = np.bincount(y[labeled_mask])
        # убираем unassigned, если он есть
        if len(unassigned_ids) > 0:
            counts[unassigned_id] = 0
        top = np.argsort(counts)[::-1][:10]
        mask = np.isin(y, top)
        print("Top labels:")
        for cls in top:
            print(f"  {cls}: {label_names[cls]}  n={counts[cls]}")

    else:
        raise ValueError(f"Unknown mode: {mode}")

    X_sel = X[mask]
    y_sel = y[mask]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_sel))
    X_sel = X_sel[perm]
    y_sel = y_sel[perm]

    return X_sel, y_sel, label_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="top10_labeled",
                        choices=["top10_labeled", "labeled", "all"])
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="checkpoints/levine13_ddpm.pt")
    args = parser.parse_args()

    device = resolve_device(args.device)
    X, y, label_names = make_dataset(args.mode)

    print("Train X:", X.shape)
    print("Device:", device)

    X_torch = torch.tensor(X, dtype=torch.float32, device=device)
    n = X_torch.shape[0]

    def sample_fn(batch_size):
        idx = torch.randint(0, n, (batch_size,), device=device)
        return X_torch[idx]

    model = DiffusionModel(
        dim=13,
        T=1000,
        beta_start=1e-4,
        beta_end=0.02,
        hidden_dims=[256, 256, 256],
        activation="silu",
        device=device,
    )

    loss_hist = model.train_on_data(
        sample_fn=sample_fn,
        num_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_min=1e-5,
        scheduler="cosine",
        log_every=1000,
        save_every=10000,
        save_path=args.out,
    )

    model.save_checkpoint(args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()