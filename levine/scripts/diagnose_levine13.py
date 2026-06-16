from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


DATA = Path("data/levine13/levine13_processed.npz")
OUT = Path("results_levine13/diagnostics")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    data = np.load(DATA, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    label_names = data["label_names"]

    print("X:", X.shape)
    print("num labels:", len(label_names))

    # Определяем unassigned, если есть
    label_names_lower = np.array([str(s).lower() for s in label_names])
    unassigned_ids = np.where(
        np.char.find(label_names_lower.astype(str), "unassigned") >= 0
    )[0]

    if len(unassigned_ids) > 0:
        unassigned_id = int(unassigned_ids[0])
        labeled_mask = y != unassigned_id
    else:
        labeled_mask = np.ones(len(y), dtype=bool)

    X_lab = X[labeled_mask]
    y_lab = y[labeled_mask]

    print("labeled:", X_lab.shape)

    # kNN purity
    k = 50
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X_lab)

    # чтобы не считать слишком долго, берём подвыборку
    rng = np.random.default_rng(42)
    m = min(20000, len(X_lab))
    idx = rng.choice(len(X_lab), size=m, replace=False)

    dist, ind = nn.kneighbors(X_lab[idx])
    neigh = ind[:, 1:]  # убрать саму точку
    y_neigh = y_lab[neigh]
    y_self = y_lab[idx][:, None]

    purity = (y_neigh == y_self).mean(axis=1)

    print(f"Mean purity@{k}: {purity.mean():.3f}")
    print(f"Median purity@{k}: {np.median(purity):.3f}")

    # purity по классам
    rows = []
    for cls in np.unique(y_lab):
        cls_mask = y_lab[idx] == cls
        if cls_mask.sum() == 0:
            continue
        rows.append({
            "label": str(label_names[cls]),
            "n_eval": int(cls_mask.sum()),
            f"mean_purity@{k}": float(purity[cls_mask].mean()),
            f"median_purity@{k}": float(np.median(purity[cls_mask])),
        })

    df = pd.DataFrame(rows).sort_values(f"mean_purity@{k}", ascending=False)
    df.to_csv(OUT / "knn_purity_by_label.csv", index=False)
    print(df.head(30))

    # PCA plot для визуальной проверки
    m_plot = min(30000, len(X_lab))
    idx_plot = rng.choice(len(X_lab), size=m_plot, replace=False)
    X_plot = X_lab[idx_plot]
    y_plot = y_lab[idx_plot]

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X_plot)

    plt.figure(figsize=(8, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=y_plot, s=2, alpha=0.5)
    plt.title("Levine13 labeled cells: PCA projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(OUT / "pca_labeled_cells.png", dpi=200)
    print("Saved:", OUT / "pca_labeled_cells.png")


if __name__ == "__main__":
    main()