import warnings
import sys
import pandas as pd
import os

# preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer

# model
from sklearn.ensemble import GradientBoostingClassifier

# metrics
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    classification_report,
)

# mlflow
import mlflow
import logging

# Import data
PATH = r"C:\Users\thiba\Documents\Projets data\202402_NLP_emotions\data"
fichier = "text.csv"


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    acc = accuracy_score(actual, pred)
    prec = precision_score(actual, pred, average="macro")
    recall = recall_score(actual, pred, average="macro")
    f1 = f1_score(actual, pred, average="macro")
    return acc, prec, recall, f1


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Data discovery
    try:
        df = pd.read_csv(os.path.join(PATH, fichier), index_col=0)
    except Exception as e:
        logger.exception("Unable to load the csv base", e)

    # Import labels
    target_labels = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise",
    }

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    # Create a new MLflow Experiment
    mlflow.set_experiment("NLP_classifier_1")

    # Define a run name for this iteration of training.
    # If not set, a unique name will be auto-generated.
    run_name = "GBC_firstmodel_3"

    # Define an artifact path that the model will be saved to.
    artifact_path = "gbc"

    # get parameters from cli
    # learning rate
    lr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    # nombre d'estimateurs
    n_estimators = int(sys.argv[2]) if len(sys.argv) > 1 else 100
    # fraction du dataset d'entraînement
    frac = float(sys.argv[3]) if len(sys.argv) > 1 else 0.1

    mlflow.autolog()

    # Start mlflow run
    with mlflow.start_run(run_name=run_name):

        # Réduction du jeu d'entraînement pour accélerer les tests
        df_r = df.sample(frac=frac)
        # X_train needs to be a dataframe not a series!
        X_r = df_r["text"]
        y_r = df_r["label"]

        # Séparation train, test
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X_r, y_r, test_size=0.2, random_state=42
        )

        # Build pipeline with vectorizer and model
        vectorizer2 = CountVectorizer()

        # GradientBoosting
        classifier = GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=lr
        )

        # Piepiline
        pipeline = make_pipeline(vectorizer2, classifier)
        pipeline.fit(X_train_r, y_train_r)

        # Predict
        y_pred_r = pipeline.predict(X_test_r)

        (acc, prec, recall, f1) = eval_metrics(y_test_r, y_pred_r)

        print(
            "GradientBoostingClassifier (lr = %f, n_estimators = %f)\
                  with dataset fraction of %f :"
            % (lr, n_estimators, frac)
        )
        print("  Accuracy : %s" % acc)
        print("  Precision : %s" % prec)
        print("  Rappel : %s" % recall)
        print("  F1 score : %s" % f1)

        # Logging parameters
        #        mlflow.log_param("learning rate", lr)
        #        mlflow.log_param("n_estimators", n_estimators)
        #        mlflow.log_param("frac", frac)

        # Logging metrics
        #        mlflow.log_metric("Accuracy", acc)
        #        mlflow.log_metric("Precision", prec)
        #        mlflow.log_metric("Rappel", recall)
        #        mlflow.log_metric("F1 score", f1)

        print("Score accuracy train : ", pipeline.score(X_train_r, y_train_r))
        print("Score accuracy test : ", pipeline.score(X_test_r, y_test_r))
        print(classification_report(y_test_r, y_pred_r))
