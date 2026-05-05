import os
import sys

import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import compute_metrics, print_metrics  # noqa: E402
from src.preprocess import preprocess  # noqa: E402
from src.utils import download_dataset, load_config, load_raw_data  # noqa: E402


def build_model(params: dict) -> XGBClassifier:
    # Construye el clasificador XGBoost con los hiperparámetros del config
    return XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=params["random_state"],
        eval_metric="logloss",
    )


def run():
    config = load_config("config.yaml")

    data_cfg = config["data"]
    model_cfg = config["model"]
    mlflow_cfg = config["mlflow"]

    # Descarga el dataset si no existe localmente
    print(f"Descargando dataset desde: {data_cfg['url']}")
    download_dataset(data_cfg["url"], data_cfg["raw_path"])

    df = load_raw_data(data_cfg["raw_path"])
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    # Preprocesamiento completo: limpieza, encoding, escalamiento y división
    X_train, X_test, y_train, y_test, scaler = preprocess(
        df=df,
        target_col=data_cfg["target_column"],
        threshold=data_cfg["quality_threshold"],
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
    )
    print(f"Entrenamiento: {X_train.shape[0]} muestras | Prueba: {X_test.shape[0]} muestras")
    print(f"Distribución de clases (train): {dict(y_train.value_counts().sort_index())}")

    # Configuración del tracking con MLflow
    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name=mlflow_cfg["run_name"]):
        # Registro de parámetros del modelo y del dataset
        mlflow.log_params(model_cfg)
        mlflow.log_params({
            "test_size": data_cfg["test_size"],
            "quality_threshold": data_cfg["quality_threshold"],
            "dataset_url": data_cfg["url"],
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
        })

        # Entrenamiento del modelo
        model = build_model(model_cfg)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Predicción y cálculo de métricas
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        print_metrics(metrics)

        mlflow.log_metrics(metrics)

        # Registro del modelo con firma y ejemplo de entrada
        input_example = X_test.iloc[:5]
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=mlflow_cfg["model_name"],
        )

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run ID: {run_id}")
        print(f"Tracking URI: {tracking_uri}")

    return metrics


if __name__ == "__main__":
    run()
