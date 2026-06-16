from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from src.models.diffusion import DiffusionModel
from src.algorithms.mode_finder_v2 import ddim_sample


OUT = Path("results_levine13/generation")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    data = np.load("data/levine13/levine13_processed.npz", allow_pickle=True)
    X = data["X"].astype(np.float32)

    model = DiffusionModel.from_checkpoint(args.ckpt, device=args.device)

    samples = ddim_sample(model, n=args.n, num_steps=50, seed=123)
    samples = np.asarray(samples, dtype=np.float32)

    # PCA на реальных данных, потом туда же samples
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(30000, len(X)), replace=False)
    X_ref = X[idx]

    pca = PCA(n_components=2, random_state=42)
    Z_real = pca.fit_transform(X_ref)
    Z_gen = pca.transform(samples)

    plt.figure(figsize=(8, 6))
    plt.scatter(Z_real[:, 0], Z_real[:, 1], s=2, alpha=0.25, label="real")
    plt.scatter(Z_gen[:, 0], Z_gen[:, 1], s=4, alpha=0.5, label="generated")
    plt.legend()
    plt.title("Levine13: real vs DDIM samples in PCA")
    plt.tight_layout()
    plt.savefig(OUT / "real_vs_generated_pca.png", dpi=200)

    # kNN distance: насколько generated близки к real manifold
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_ref)
    dist_gen, _ = nn.kneighbors(samples)

    idx2 = rng.choice(len(X_ref), size=min(args.n, len(X_ref)), replace=False)
    dist_real, _ = nn.kneighbors(X_ref[idx2])

    print("Mean nearest real distance for generated:", float(dist_gen.mean()))
    print("Mean nearest real distance for real subset:", float(dist_real.mean()))
    print("Saved:", OUT / "real_vs_generated_pca.png")


if __name__ == "__main__":
    main()