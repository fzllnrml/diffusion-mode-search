from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = Path("data/levine13")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARKERS = [
    "CD45", "CD45RA", "CD19", "CD11b", "CD4", "CD8", "CD34",
    "CD20", "CD33", "CD123", "CD38", "CD90", "CD3"
]


def load_dataframe():
    txt_candidates = [
        DATA_DIR / "Levine_13dim.txt",
        DATA_DIR / "Levine_13dim.tsv",
    ]

    for path in txt_candidates:
        if path.exists():
            print(f"Reading TXT/TSV: {path}")
            return pd.read_csv(path, sep="\t")

    fcs_path = DATA_DIR / "Levine_13dim.fcs"
    if fcs_path.exists():
        print(f"Reading FCS: {fcs_path}")
        from fcsparser import parse
        meta, df = parse(str(fcs_path), reformat_meta=True)
        return df

    raise FileNotFoundError(
        "Не нашёл Levine_13dim.txt/.tsv или Levine_13dim.fcs в data/"
    )


def find_marker_columns(df):
    cols = list(df.columns)

    marker_cols = []
    for marker in MARKERS:
        matches = [c for c in cols if str(c).strip() == marker]
        if not matches:
            matches = [c for c in cols if marker.lower() in str(c).lower()]
        if matches:
            marker_cols.append(matches[0])

    if len(marker_cols) != 13:
        print("Columns:")
        for c in cols:
            print("  ", c)
        raise RuntimeError(
            f"Нашёл только {len(marker_cols)} marker columns вместо 13: {marker_cols}"
        )

    return marker_cols


def find_label_column(df):
    cols = list(df.columns)
    candidates = []

    for c in cols:
        cl = str(c).lower()
        if any(word in cl for word in ["label", "population", "cluster", "cell_type"]):
            candidates.append(c)

    print("Possible label columns:", candidates)

    if not candidates:
        print("Не нашёл label column автоматически.")
        print("Все columns:")
        for c in cols:
            print("  ", c)
        return None

    # Часто надо будет вручную поправить после просмотра columns.
    return candidates[0]


def main():
    df = load_dataframe()
    print("Loaded shape:", df.shape)
    print("First columns:", list(df.columns)[:30])

    marker_cols = find_marker_columns(df)
    label_col = find_label_column(df)

    print("Marker columns:", marker_cols)
    print("Label column:", label_col)

    X = df[marker_cols].to_numpy(dtype=np.float32)

    # labels могут быть строками, числами или NaN
    if label_col is not None:
        labels_raw = df[label_col].astype("object").where(df[label_col].notna(), "unassigned")
        labels_str = labels_raw.astype(str).to_numpy()
    else:
        labels_str = np.array(["unknown"] * len(X), dtype=object)

    # Убираем строки с inf/nan в признаках
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    labels_str = labels_str[mask]

    # StandardScaler нужен даже после arcsinh
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Кодируем labels в integers
    unique_labels, y = np.unique(labels_str, return_inverse=True)

    np.savez_compressed(
        OUT_DIR / "levine13_processed.npz",
        X=X_scaled,
        X_raw=X.astype(np.float32),
        y=y.astype(np.int64),
        label_names=unique_labels.astype(str),
        marker_names=np.array(MARKERS, dtype=str),
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
    )

    print("Saved:", OUT_DIR / "levine13_processed.npz")
    print("X_scaled:", X_scaled.shape)
    print("Labels:")
    counts = pd.Series(labels_str).value_counts()
    print(counts.head(30))


if __name__ == "__main__":
    main()
