import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import (  # noqa: E402
    clean_data,
    create_binary_target,
    scale_features,
    split_features_target,
)
from src.evaluate import compute_metrics  # noqa: E402
from src.utils import load_config  # noqa: E402


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "fixed acidity": [7.4, 7.8, 7.8, 11.2, 7.4],
        "volatile acidity": [0.70, 0.88, 0.76, 0.28, 0.66],
        "citric acid": [0.00, 0.00, 0.04, 0.56, 0.00],
        "residual sugar": [1.9, 2.6, 2.3, 1.9, 1.8],
        "chlorides": [0.076, 0.098, 0.092, 0.075, 0.075],
        "free sulfur dioxide": [11.0, 25.0, 15.0, 17.0, 13.0],
        "total sulfur dioxide": [34.0, 67.0, 54.0, 60.0, 40.0],
        "density": [0.9978, 0.9968, 0.9970, 0.9980, 0.9978],
        "pH": [3.51, 3.20, 3.26, 3.16, 3.51],
        "sulphates": [0.56, 0.68, 0.65, 0.58, 0.56],
        "alcohol": [9.4, 9.8, 9.8, 9.8, 9.4],
        "quality": [5, 5, 5, 6, 5],
    })


def test_clean_data_removes_duplicates(sample_df):
    df_with_dup = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    cleaned = clean_data(df_with_dup)
    assert len(cleaned) == len(sample_df)


def test_clean_data_removes_nulls():
    df = pd.DataFrame({"a": [1, None, 3], "quality": [5, 6, 7]})
    cleaned = clean_data(df)
    assert cleaned.isnull().sum().sum() == 0


def test_clean_data_normalizes_columns():
    df = pd.DataFrame({"fixed acidity": [7.4], "quality": [5]})
    cleaned = clean_data(df)
    assert "fixed_acidity" in cleaned.columns


def test_create_binary_target(sample_df):
    cleaned = clean_data(sample_df)
    labeled = create_binary_target(cleaned, "quality", threshold=6)
    assert "label" in labeled.columns
    assert "quality" not in labeled.columns
    assert set(labeled["label"].unique()).issubset({0, 1})


def test_binary_target_threshold():
    df = pd.DataFrame({"quality": [4, 5, 6, 7, 8]})
    result = create_binary_target(df, "quality", threshold=6)
    expected = [0, 0, 1, 1, 1]
    assert list(result["label"]) == expected


def test_split_features_target(sample_df):
    cleaned = clean_data(sample_df)
    labeled = create_binary_target(cleaned, "quality", threshold=6)
    X, y = split_features_target(labeled)
    assert "label" not in X.columns
    assert len(X) == len(y)
    assert X.shape[1] == 11


def test_scale_features_zero_mean():
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    X_test = pd.DataFrame({"a": [1.0, 3.0, 5.0]})
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
    assert abs(X_train_s["a"].mean()) < 1e-9


def test_compute_metrics_perfect_prediction():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.9, 0.1, 0.9])
    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_metrics_keys():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.4, 0.6])
    metrics = compute_metrics(y_true, y_pred, y_prob)
    for key in ("accuracy", "f1_score", "roc_auc", "precision", "recall"):
        assert key in metrics


def test_load_config_structure():
    config = load_config("config.yaml")
    assert "data" in config
    assert "model" in config
    assert "mlflow" in config
    assert "url" in config["data"]
    assert "n_estimators" in config["model"]


def test_config_data_values():
    config = load_config("config.yaml")
    assert 0 < config["data"]["test_size"] < 1
    assert config["data"]["quality_threshold"] >= 1
    assert config["model"]["n_estimators"] > 0
