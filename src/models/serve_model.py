import mlflow

# import from registry
model_name = "Sentiment_classifier_GBC"
model_version = 4
model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")

model.serve(port=5000)
