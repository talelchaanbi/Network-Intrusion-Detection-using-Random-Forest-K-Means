"""
Network Intrusion Detection with Random Forest & K-Means (Kaggle-ready)

Usage on Kaggle:
1) Add NSL-KDD dataset in notebook inputs.
2) Run this script.
3) Outputs:
   - metrics in console
   - resultats.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.io import arff


NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]


def find_dataset() -> Path:
    root = Path("/kaggle/input")
    if not root.exists():
        raise FileNotFoundError("/kaggle/input introuvable")

    all_files = [p for p in root.rglob("*") if p.is_file()]

    print("Exemples de fichiers trouvés:")
    for p in all_files[:40]:
        print(" -", p)

    train_files = [
        p
        for p in all_files
        if ("kdd" in p.name.lower())
        and ("train" in p.name.lower())
        and (p.suffix.lower() in {".txt", ".csv", ".arff"})
    ]

    if not train_files:
        raise FileNotFoundError(
            "Aucun fichier KDDTrain trouvé (.txt/.csv/.arff). "
            "Ajoute le dataset NSL-KDD via Add Input > Datasets."
        )

    return train_files[0]


def find_train_test_files() -> tuple[Path, Path | None]:
    root = Path("/kaggle/input")
    if not root.exists():
        raise FileNotFoundError("/kaggle/input introuvable")

    all_files = [p for p in root.rglob("*") if p.is_file()]

    train_candidates = [
        p
        for p in all_files
        if ("kdd" in p.name.lower())
        and ("train" in p.name.lower())
        and (p.suffix.lower() in {".txt", ".csv", ".arff"})
    ]
    test_candidates = [
        p
        for p in all_files
        if ("kdd" in p.name.lower())
        and ("test" in p.name.lower())
        and (p.suffix.lower() in {".txt", ".csv", ".arff"})
    ]

    if not train_candidates:
        raise FileNotFoundError(
            "Aucun fichier KDDTrain trouvé (.txt/.csv/.arff). "
            "Ajoute le dataset NSL-KDD via Add Input > Datasets."
        )

    def _rank(p: Path) -> tuple[int, int, int]:
        name = p.name.lower()
        is_full_train = 0 if "kddtrain+.txt" == name else 1
        is_txt = 0 if p.suffix.lower() == ".txt" else 1
        is_percent = 1 if "20percent" in name else 0
        return (is_full_train, is_txt, is_percent)

    def _rank_test(p: Path) -> tuple[int, int, int]:
        name = p.name.lower()
        is_full_test = 0 if "kddtest+.txt" == name else 1
        is_txt = 0 if p.suffix.lower() == ".txt" else 1
        is_minus21 = 1 if "-21" in name else 0
        return (is_full_test, is_txt, is_minus21)

    train_path = sorted(train_candidates, key=_rank)[0]
    test_path = sorted(test_candidates, key=_rank_test)[0] if test_candidates else None
    return train_path, test_path


def load_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".arff":
        data, _ = arff.loadarff(str(path))
        df = pd.DataFrame(data)

        # bytes -> str
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].apply(
                    lambda v: v.decode("utf-8")
                    if isinstance(v, (bytes, bytearray))
                    else v
                )

        if df.shape[1] == 43:
            df.columns = NSL_KDD_COLUMNS

        if "label" not in df.columns:
            raise ValueError("Colonne 'label' non trouvée dans le fichier ARFF.")
        return df

    # txt/csv
    try:
        df = pd.read_csv(path, header=None)
        if df.shape[1] == 43:
            df.columns = NSL_KDD_COLUMNS
            return df
    except Exception:
        pass

    df = pd.read_csv(path)
    if "label" not in df.columns and df.shape[1] == 43:
        df.columns = NSL_KDD_COLUMNS

    if "label" not in df.columns:
        raise ValueError("Colonne 'label' non trouvée. Dataset NSL-KDD invalide.")

    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = (df["label"] != "normal").astype(int)
    x = df.drop(columns=[c for c in ["label", "difficulty"] if c in df.columns]).copy()

    cat_cols = x.select_dtypes(include=["object"]).columns.tolist()
    x = pd.get_dummies(x, columns=cat_cols, drop_first=True)
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0)
    return x, y


def main() -> None:
    print("[1/6] Recherche dataset...")
    train_path, test_path = find_train_test_files()
    print(f"Train dataset: {train_path}")
    if test_path is not None:
        print(f"Test dataset: {test_path}")

    print("[2/6] Chargement...")
    train_df = load_data(train_path)
    print(f"Train shape brut: {train_df.shape}")
    test_df = load_data(test_path) if test_path is not None else None
    if test_df is not None:
        print(f"Test shape brut: {test_df.shape}")

    print("[3/6] Prétraitement...")
    x_train_full, y_train_full = preprocess(train_df)
    if test_df is not None:
        x_test_full, y_test_full = preprocess(test_df)
        x_test_full = x_test_full.reindex(columns=x_train_full.columns, fill_value=0)
        x_for_kmeans = pd.concat([x_train_full, x_test_full], axis=0)
        y_for_kmeans = pd.concat([y_train_full, y_test_full], axis=0)
        print(f"Train features: {x_train_full.shape}")
        print(f"Test features: {x_test_full.shape}")
    else:
        x_for_kmeans = x_train_full
        y_for_kmeans = y_train_full
        print(f"Shape features: {x_train_full.shape}")

    print("[4/6] Random Forest...")
    if test_df is not None:
        x_train, y_train = x_train_full, y_train_full
        x_test, y_test = x_test_full, y_test_full
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )

    rf = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    print("\n===== Random Forest =====")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["normal", "attack"]))

    print("[5/6] K-Means...")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_for_kmeans)

    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = km.fit_predict(x_scaled)

    sil = silhouette_score(x_scaled, clusters)
    print("\n===== K-Means =====")
    print(f"Silhouette Score: {sil:.4f}")

    # Mapping approximatif cluster -> label (majority vote)
    mapping = {}
    for c in np.unique(clusters):
        mask = clusters == c
        mapping[c] = int(pd.Series(y_for_kmeans[mask]).mode().iloc[0])

    mapped = np.vectorize(mapping.get)(clusters)
    approx_acc = accuracy_score(y_for_kmeans, mapped)
    print(f"Approx cluster-label accuracy: {approx_acc:.4f}")

    print("[6/6] Sauvegarde figure...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0], colorbar=False)
    axes[0].set_title("Random Forest - Confusion Matrix")

    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(x_scaled)
    scatter = axes[1].scatter(
        x_pca[:, 0], x_pca[:, 1], c=clusters, cmap="viridis", s=8, alpha=0.6
    )
    axes[1].set_title("K-Means Clusters (PCA 2D)")
    axes[1].set_xlabel("PCA 1")
    axes[1].set_ylabel("PCA 2")
    fig.colorbar(scatter, ax=axes[1], label="Cluster")

    plt.tight_layout()
    plt.savefig("resultats.png", dpi=200)
    plt.close()

    print("✅ Terminé. Fichier généré: resultats.png")


if __name__ == "__main__":
    main()
