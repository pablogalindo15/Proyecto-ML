"""
train_v7.py — Pipeline ganador F1 = 0.2906
===========================================
Config exacta de train_v6 Fase 2:
  vectorizador : char(2,5)  max_features=300k
  stack        : LR_lbfgs(C=16) + LR_liblinear(C=7)
  meta         : LR(C=1)  |  3-fold
  sin OCR  |  sin dedup  |  sin balanced

Fases:
  1. Validación 80/20  → imprime F1
  2. Reentrenamiento con el 100% de train
  3. Exporta submission_alt.csv  y  modelo_alt.joblib
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE  = 42
EXPECTED_F1   = 0.2906
np.random.seed(RANDOM_STATE)

BASE_DIR         = Path(__file__).parent
TRAIN_PATH       = BASE_DIR / "Data" / "train.csv"
EVAL_PATH        = BASE_DIR / "Data" / "eval.csv"
SUBMISSION_PATH  = BASE_DIR / "submission_alt.csv"
MODEL_PATH       = BASE_DIR / "modelo_alt.joblib"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_train():
    df = pd.read_csv(TRAIN_PATH)
    df = df.dropna(subset=["text", "decade"])
    df = df[df["text"].str.strip().ne("")]
    df["decade"] = df["decade"].astype(int)
    return df["text"], df["decade"]


def load_eval():
    df = pd.read_csv(EVAL_PATH)
    df["text"] = df["text"].fillna("")
    return df


def build_pipeline() -> Pipeline:
    """Pipeline ganador exacto de train_v6 Fase 2."""
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        max_features=300_000,
        lowercase=True,
    )
    stack = StackingClassifier(
        estimators=[
            ("lr_lbfgs", LogisticRegression(
                solver="lbfgs", C=16.0,
                max_iter=2000, random_state=RANDOM_STATE,
            )),
            ("lr_liblinear", LogisticRegression(
                solver="liblinear", C=7.0, penalty="l2", dual=True,
                max_iter=2000, random_state=RANDOM_STATE,
            )),
        ],
        final_estimator=LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_STATE,
        ),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        verbose=1,
    )
    return Pipeline([("vec", vec), ("stack", stack)])


# ── 1. Validación 80/20 ───────────────────────────────────────────────────────

def validate(X, y):
    print("\n── 1. Validación 80/20 ──────────────────────────────────────────────")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}")

    pipe = build_pipeline()
    t0 = time.time()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="macro")
    elapsed = (time.time() - t0) / 60

    print(f"\n  Macro F1 val : {f1:.4f}  (esperado ≈ {EXPECTED_F1})")
    delta = f1 - EXPECTED_F1
    print(f"  Δ esperado   : {delta:+.4f}")
    print(f"  Tiempo       : {elapsed:.1f} min")
    print()
    print(classification_report(y_val, y_pred))
    return f1


# ── 2. Reentrenamiento final + exportación ────────────────────────────────────

def train_and_export(X, y, eval_df):
    print("\n── 2. Reentrenamiento final (100% train) ────────────────────────────")
    pipe = build_pipeline()
    t0 = time.time()
    pipe.fit(X, y)
    elapsed = (time.time() - t0) / 60
    print(f"  Tiempo reentrenamiento: {elapsed:.1f} min")

    # Modelo
    joblib.dump(pipe, MODEL_PATH)
    size_mb = MODEL_PATH.stat().st_size / (1024 ** 2)
    print(f"  Modelo guardado: {MODEL_PATH}  ({size_mb:.1f} MB)")

    # Submission
    y_eval = pipe.predict(eval_df["text"])
    submission = pd.DataFrame({"id": eval_df["id"], "decade": y_eval.astype(int)})
    assert len(submission) == len(eval_df)
    assert submission.isna().sum().sum() == 0
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"  Submission guardada: {SUBMISSION_PATH}  ({len(submission)} filas)")
    print(f"\n  {submission.head().to_string(index=False)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()
    print("=" * 60)
    print("  train_v7.py — Config ganadora F1 ≈ 0.2906")
    print("  char(2,5) 300k | Stack LR_lbfgs(C=16) + LR_lib(C=7)")
    print("  Exporta: submission_alt.csv  |  modelo_alt.joblib")
    print("=" * 60)

    X, y = load_train()
    eval_df = load_eval()
    print(f"\n  Train: {len(X)} filas  |  Eval: {len(eval_df)} filas")

    val_f1 = validate(X, y)
    train_and_export(X, y, eval_df)

    elapsed = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"  DONE — Tiempo total : {elapsed:.1f} min")
    print(f"  F1 validación      : {val_f1:.4f}")
    print(f"  Archivos generados : {SUBMISSION_PATH.name} | {MODEL_PATH.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
