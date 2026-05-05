from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    # Calcula las cinco métricas de evaluación del modelo
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }


def print_metrics(metrics: dict) -> None:
    print("\n--- Resultados de evaluación ---")
    for name, value in metrics.items():
        print(f"  {name:<12}: {value:.4f}")
    print("--------------------------------\n")
