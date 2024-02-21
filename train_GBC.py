import os
import warnings 
import sys 
import pandas as pd 
import numpy as np

#preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#model
from sklearn.ensemble import GradientBoostingClassifier
#metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

#mlflow
import mlflow
import logging


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    acc = accuracy_score(actual, pred)
    prec = precision_score(actual, pred, average = "macro")
    recall = recall_score(actual, pred, average = "macro")
    f1 = f1_score(actual, pred, average = "macro")
    return acc, prec, recall, f1

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    #Data discovery
    fichier = ("./data/text.csv")
    try:
        df = pd.read_csv(fichier, index_col = 0)
    except Exception as e:
        logger.exception(
            "Unable to load the csv base", e
        )

    #Import labels
    target_labels = {
        0 : "sadness",
        1 : "joy",
        2 : "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }


    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


    # Create a new MLflow Experiment
    mlflow.set_experiment("NLP_classifier_1")


    #parameters
    lr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    n_estimators = int(sys.argv[2]) if len(sys.argv) > 1 else 100
    frac = float(sys.argv[3]) if len(sys.argv) > 1 else 0.1



    #Réduction du jeu d'entraînement pour accélerer les tests
    df_r = df.sample(frac = frac)
    df_r.shape
    X_r = df_r["text"]
    y_r = df_r["label"]
    #Séparation train, test
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size = 0.2, random_state = 42)

    #Preprocessing vectorisation
    vectorizer2 = CountVectorizer()
    X_train_r = vectorizer2.fit_transform(X_train_r)
    X_test_r = vectorizer2.transform(X_test_r)

    # Define a run name for this iteration of training.
    # If this is not set, a unique name will be auto-generated for your run.
    run_name = "GBC_firstmodel"

    # Define an artifact path that the model will be saved to.
    artifact_path = "gbc"


    #Start mlflow run
    with mlflow.start_run(run_name = run_name):

        #GradientBoosting
        classifier = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=lr)
        classifier.fit(X_train_r, y_train_r)
        y_pred_r = classifier.predict(X_test_r)


        (acc, prec, recall, f1) = eval_metrics(y_test_r, y_pred_r)

        print("GradientBoostingClassifier (lr = %f, n_estimators = %f) with dataset fraction of %f :" % (lr, n_estimators, frac))
        print("  Accuracy : %s" % acc)
        print("  Precision : %s" % prec)
        print("  Rappel : %s" % recall)
        print("  F1 score : %s" % f1)

        #Logging parameters
        mlflow.log_param("learning rate", lr)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("frac", frac)

        #Logging metrics
        mlflow.log_metric("Accuracy", acc)
        mlflow.log_metric("Precision", prec)
        mlflow.log_metric("Rappel", recall)
        mlflow.log_metric("F1 score", f1)
        
        #Log model
        mlflow.sklearn.log_model(sk_model=classifier, input_example=X_train_r, artifact_path=artifact_path)

        print("Score accuracy train : ", classifier.score(X_train_r, y_train_r))
        print("Score accuracy test : ", classifier.score(X_test_r, y_test_r))
        print(classification_report(y_test_r, y_pred_r))

    
