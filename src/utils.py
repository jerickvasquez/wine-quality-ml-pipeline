import os
import yaml
import requests
import pandas as pd


def load_config(path: str = "config.yaml") -> dict:
    # Carga la configuración desde el archivo YAML
    with open(path, "r") as f:
        return yaml.safe_load(f)


def download_dataset(url: str, dest_path: str) -> None:
    # Descarga el dataset solo si no existe localmente
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        return
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(response.content)


def load_raw_data(path: str) -> pd.DataFrame:
    # El dataset usa punto y coma como separador
    return pd.read_csv(path, sep=";")
