# Credit Risk Model

## Introducción
Proyecto para comparar los resulatdos de varias familias de modelos.

## Requerimientos
* poetry
* python version ^3.9

## Estructura del proyecto
```
.
└── credit_risk_model
    ├── Dockerfile
    ├── README.md
    ├── data
        ├── preprocessed
        └── raw
    ├── models
    ├── poetry.lock
    ├── pyproject.toml
    ├── src
    │    ├── feature_engineering
    │    │  ├── 00.create_dataframe.py
    │    │  ├── 01.clean_data.py
    │    │  ├── __init__.py
    │    │  ├── constants.py
    │    │  ├── new_features.py
    │    │  └── utils.py
    │    └── work_model
    │       ├── 02.credit_scoring.py
    │       ├── 03.neural_network.py
    │       ├── 04.neural_network.py
    │       └── dic_models.py
    └── tests


```

## Instalación

En le directorio del proyecto abrimos un terminal y ejecutamos el comando `poetry install` para descargar e instalar todas las librerias necesarias en el entorno virtual que poetry generará automáticamente.

## Uso

En el directorio del proyecto ejecutar el comando `poetry shell` para activar el entorno virtual con las dependencias instaladas.
Según la tipología del modelo que se quiera ejectuar se deberán lanzar los siguientes comandos que devolverán el resultado por pantalla:

### AD

### LR

### SVM

### RF

### CB

### LRN

### MLP

### ResNet


