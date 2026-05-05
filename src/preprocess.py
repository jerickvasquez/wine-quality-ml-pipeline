import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Normaliza nombres de columnas, elimina nulos y duplicados
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df = df.dropna()
    df = df.drop_duplicates()
    return df


def create_binary_target(df: pd.DataFrame, target_col: str, threshold: int) -> pd.DataFrame:
    # Convierte la calidad en clasificación binaria: 1 = bueno, 0 = no bueno
    df = df.copy()
    df["label"] = (df[target_col] >= threshold).astype(int)
    df = df.drop(columns=[target_col])
    return df


def split_features_target(df: pd.DataFrame, label_col: str = "label"):
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    # Ajusta el scaler solo con datos de entrenamiento para evitar data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    return X_train_scaled, X_test_scaled, scaler


def preprocess(df: pd.DataFrame, target_col: str, threshold: int, test_size: float, random_state: int):
    # Pipeline completo de preprocesamiento
    df = clean_data(df)
    df = create_binary_target(df, target_col, threshold)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_test, scaler = scale_features(X_train, X_test)
    return X_train, X_test, y_train, y_test, scaler
