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
    ├── data
    │   ├── preprocessed /* carpeta donde se guarda el dato raw */
    │   └── raw /* carpeta donde se guarda el dato preprocessed */
    ├── models /* carpeta donde se guardan los modelos */
    ├── src
    │    ├── commons
    │    ├── feature_engineering /* donde se encuentran los códigos de limpieza de datos */
    └──  └── work_model /* dónde se encuentran los códigos para ejecutar los modelos */

```

## Instalación

En le directorio del proyecto abrimos un terminal y ejecutamos el comando `poetry install with --dev` para descargar e
instalar todas las librerias necesarias en el entorno virtual que poetry generará automáticamente.

En el directorio del proyecto ejecutar el comando `poetry shell` para activar el entorno virtual con las dependencias instaladas.
Según la tipología del modelo que se quiera ejectuar se deberán lanzar los siguientes comandos que devolverán el resultado por pantalla:

## Uso

### Carga de datos
Los datos compartidos en la carpeta de drive  `TFM Irene Casas` se cargan en data/preprocessed y data/raw

### Limpieza de datos
Para ejecutar la parte de la limpieza de datos el fichero de python esta en `src/feature_engineering/01.clean_data.py`.
Esta parte solo es necesaria ejectuar si se quiere partir de los datos raw, para conseguir los datos preprocessed,
en caso contrario, no sería necesaria, pues los datos preprocessed ya tienen todas estas transformaciones realizadas.

### Modelos

### LDA

Para ejecutar los resultados presentados en la memoria del TFM ejecutar: `python src/main.py --model_type lda`

### LR

Para ejecutar los resultados presentados en la memoria del TFM ejecutar: `python src/main.py --model_type lr`

### SVM

Para ejecutar los resultados presentados en la memoria del TFM ejecutar: `python src/main.py --model_type svm`
Advertencia: la carga de este modelo es muy lenta.

### RF

Para ejecutar los resultados presentados en la memoria del TFM ejecutar: `python src/main.py --model_type rf`

### CB

Para ejecutar los resultados presentados en la memoria del TFM ejecutar: `python src/main.py --model_type cb`

### LRN

La ejecución del siguiente comando supone el entrenamiento de nuevo del modelo, pues el objeto no fue guardado.
Para ejecutar los resultados presentados en la memoria del TFM ejecutar: `python src/main.py --model_type lrn`

### MLP

La ejecución del siguiente comando supone el entrenamiento de nuevo del modelo, pues el objeto no fue guardado.
Para ejecutar los resultados presentados en la memoria del TFM ejecutar: `python src/main.py --model_type mlp`

### ResNet

La ejecución del siguiente comando supone el entrenamiento de nuevo del modelo, pues el objeto no fue guardado.
Para ejecutar los resultados presentados en la memoria del TFM ejecutar: `python src/main.py --model_type res`

