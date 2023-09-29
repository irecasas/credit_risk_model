import argparse

from work_model.catboost import result_CB
from work_model.lda import result_LDA
from work_model.logistic_regresion import result_LR
from work_model.lr_network import result_lrn
from work_model.mlp import result_mlp
from work_model.random_forest import result_RF
from work_model.resnet import result_rn
from work_model.svm import result_SVM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, help="Family model to execute")

    args = parser.parse_args()

    if args.model_type == "lda":
        result_LDA()

    if args.model_type == "lr":
        result_LR()

    if args.model_type == "svm":
        result_SVM()

    if args.model_type == "rf":
        result_RF()

    if args.model_type == "cb":
        result_CB()

    if args.model_type == "lrn":
        result_lrn()

    if args.model_type == "mlp":
        result_mlp()

    if args.model_type == "rn":
        result_rn()
