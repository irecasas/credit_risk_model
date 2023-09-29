import os

DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data")
)
DATA_PATH_PREPROCESSED = os.path.abspath(
    os.path.join(DATA_PATH, "preprocessed")
)
DATA_PATH_RAW = os.path.abspath(
    os.path.join(DATA_PATH, "raw")
)

MODELS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models")
)
