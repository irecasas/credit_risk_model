# DS Model Fraud Risk
Repositorio centralizado de feature engineering para los modelos.

## Requirements
* poetry
* docker
* python version 3.10.12

## Project Structure
```
.
└── ds-model-fraud-risk
    ├── Dockerfile
    ├── README.md
    ├── data
    │   ├── preprocessed
    │   │   └── sample_data.csv
    │   └── raw
    │       └── sample_data.csv
    ├── notebooks
    │   └── sample.ipynb
    ├── pyproject.toml
    ├── src
    │   └── ds_core
    │       └── preprocessing
    │           └── process.py
    └── tests
        └── test_preprocessing
            └── test_process.py
```

## Setup
https://confluence.publishing.tools/display/ResD/Project+Environment+Set+Up+-+Pyenv+and+Poetry
* `poe init`

### Tests

#### Poe
`poe test`

####  Docker
Setup requires that you already have docker installed on your machine.

Build the image with `docker build -t ds-model-fraud-risk .`

## Additional resources
TODO

## Contact
* irene.casas@aplazame.com