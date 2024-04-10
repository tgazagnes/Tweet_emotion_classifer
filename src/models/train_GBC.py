import warnings
import sys
import pandas as pd
import os

# preprocessing
from sklearn.model_selection import train_test_split
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

# functions to create model wrapper and include preprocessing function
from mlflow.pyfunc import PythonModel, PythonModelContext
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

    # parameters
    # learning rate
    lr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    # nombre d'estimateurs
    n_estimators = int(sys.argv[2]) if len(sys.argv) > 1 else 100
    # fraction du dataset d'entraînement
    frac = float(sys.argv[3]) if len(sys.argv) > 1 else 0.1

    # Réduction du jeu d'entraînement pour accélerer les tests
    df_r = df.sample(frac=frac)
    df_r.shape
    X_r = df_r["text"]
    y_r = df_r["label"]
    # Séparation train, test
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_r, y_r, test_size=0.2, random_state=42
    )

    # Preprocessing vectorisation
    vectorizer2 = CountVectorizer()
    X_train_r = vectorizer2.fit_transform(X_train_r)
    X_test_r = vectorizer2.transform(X_test_r)

    ######################################

    # Create a model wrapper to include the preprocessing step
    # Source : https://learn.microsoft.com/en-us/azure\
    #   /machine-learning/how-to-log-mlflow-models?view=azureml-api-2&tabs=wrapper

    class ModelWrapper(PythonModel):
        def __init__(self, model):
            self.model = model

        def predict(self, context: PythonModelContext, data):
            data = vectorizer2.transform(data)
            return self.model.predict(data)

    ######################################

    # Define a run name for this iteration of training.
    # If not set, a unique name will be auto-generated.
    run_name = "GBC_firstmodel_2"

    # Define an artifact path that the model will be saved to.
    artifact_path = "gbc"

    # Start mlflow run
    with mlflow.start_run(run_name=run_name):
        # GradientBoosting
        classifier = GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=lr
        )
        classifier.fit(X_train_r, y_train_r)
        y_pred_r = classifier.predict(X_test_r)

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
        mlflow.log_param("learning rate", lr)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("frac", frac)

        # Logging metrics
        mlflow.log_metric("Accuracy", acc)
        mlflow.log_metric("Precision", prec)
        mlflow.log_metric("Rappel", recall)
        mlflow.log_metric("F1 score", f1)

        # Log encoder & model
        #        vectorizer_path = "vectorizer.pkl"
        #        joblib.dump(vectorizer2, vectorizer_path)
        #        model_path = "gbclassifier.model"
        #        classifier.save_model(model_path)

        # record signature
        # Model signature defines the expected format for model inputs and outputs
        # including any additional parameters needed for inference.
        signature = mlflow.models.infer_signature(
            X_r, classifier.predict(vectorizer2.transform(X_r))
        )

        # Including an input example while logging a model offers benefit
        # https://mlflow.org/docs/latest/model/signatures.html#input-example

        # custom model to include preprocessing (vectorizer)
        mlflow.pyfunc.log_model(
            "GBClassifier",
            python_model=ModelWrapper(classifier),
            #                               artifacts = {
            #                                   "vectorizer" : vectorizer_path,
            #                                   "model" : model_path
            #                               },
            signature=signature,
        )

        print("Score accuracy train : ", classifier.score(X_train_r, y_train_r))
        print("Score accuracy test : ", classifier.score(X_test_r, y_test_r))
        print(classification_report(y_test_r, y_pred_r))
