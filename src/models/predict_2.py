import mlflow
import pandas as pd
import os

# import Logged model from MLFlow experiments
# logged_model = 'runs:/a9357f870eab497b98d58150ecfb108f/GBClassifier'
# Load model as a PyFuncModel.
# loaded_model  = mlflow.pyfunc.load_model(logged_model)


# import from registry
model_name = "Sentiment_classifier_GBC"
model_version = 3
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")


# Import test data
PATH = r"C:\Users\thiba\Documents\Projets data\202402_NLP_emotions\data"
fichier = "test_samples/Echantillon_de_test_0.csv"
df2 = pd.read_csv(os.path.join(PATH, fichier), index_col=0)
df2 = df2.drop("label", axis=1)


# Test on 1 tweet
# data = ["I am so sad"]
# df = pd.DataFrame({"text" : data})
# print(model.predict(df))

# Test on sample
pred = model.predict(df2)
df2["Prédiction"] = pred
df2.to_csv(os.path.join(PATH, "test_predictions", "Prediction.csv"))
