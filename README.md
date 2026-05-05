# Pipeline de Clasificación de Calidad de Vino

Pipeline automatizado de machine learning para clasificar la calidad del vino tinto usando XGBoost, con seguimiento de experimentos mediante MLflow y CI/CD con GitHub Actions.

## Dataset

**Red Wine Quality** — UCI Machine Learning Repository  
Fuente: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv  
Formato: CSV con separador punto y coma, 1599 filas, 12 columnas  
Tarea: Clasificación binaria — vinos con calidad >= 6 se etiquetan como "bueno" (1), los demás como "no bueno" (0)

Características: acidez fija, acidez volátil, ácido cítrico, azúcar residual, cloruros, dióxido de azufre libre, dióxido de azufre total, densidad, pH, sulfatos, alcohol

## Estructura del Proyecto

```
.
├── .github/
│   └── workflows/
│       └── ml.yml          # Pipeline de CI/CD
├── src/
│   ├── train.py            # Script principal de entrenamiento
│   ├── preprocess.py       # Limpieza y preparación de datos
│   ├── evaluate.py         # Cálculo de métricas
│   └── utils.py            # Carga de configuración y descarga del dataset
├── tests/
│   └── test_pipeline.py    # Pruebas de validación
├── config.yaml             # Hiperparámetros y rutas
├── requirements.txt
├── Makefile
└── README.md
```

## Requisitos

- Python 3.11+
- pip

## Instalación

```bash
make install
```

## Uso

### Ejecutar el pipeline completo

```bash
make train
```

O directamente:

```bash
python src/train.py
```

### Ejecutar pruebas

```bash
make test
```

### Verificar estilo del código

```bash
make lint
```

### Ejecutar todo

```bash
make all
```

## Seguimiento con MLflow

El pipeline registra automáticamente:
- **Parámetros**: hiperparámetros del modelo y configuración del dataset
- **Métricas**: accuracy, F1-score, ROC-AUC, precision, recall
- **Modelo**: XGBoost con firma de entrada y ejemplo de datos
- **Registro**: modelo registrado bajo el nombre `wine-quality-xgb`

### Ver la interfaz de MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Luego abrir http://localhost:5000 en el navegador.

## Modelo

**Algoritmo**: XGBoost Classifier  
**Tarea**: Clasificación binaria (vino bueno: calidad >= 6)

| Métrica   | Valor típico |
|-----------|--------------|
| Accuracy  | ~0.76        |
| F1-Score  | ~0.77        |
| ROC-AUC   | ~0.83        |
| Precision | ~0.78        |
| Recall    | ~0.76        |

## Configuración

Editar `config.yaml` para cambiar hiperparámetros o rutas del dataset:

```yaml
model:
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
```

## CI/CD con GitHub Actions

El flujo en `.github/workflows/ml.yml` se ejecuta automáticamente en cada push a `main` o `master`:

1. `make install` — instala dependencias
2. `make lint` — verifica estilo del código
3. `make test` — ejecuta pruebas de validación
4. `make train` — entrena y registra el modelo
5. Sube `mlflow.db` y `mlruns/` como artefactos del workflow
