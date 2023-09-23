# Credit Risk Model
Repository containing the study conducted on the comparison of different credit rating modelling.

## Requirements
* poetry
* docker
* python version 3.10.12

## Project Structure
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

## Setup
https://confluence.publishing.tools/display/ResD/Project+Environment+Set+Up+-+Pyenv+and+Poetry
* `poe init`

### Tests

#### Poe
`poe test`

####  Docker
Setup requires that you already have docker installed on your machine.

Build the image with `docker build -t credit_risk_model .`

## Additional resources
TODO

