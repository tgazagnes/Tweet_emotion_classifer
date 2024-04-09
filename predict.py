import mlflow.pyfunc
import mlflow
import pandas as pd

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

model_name = "GBC_classifier_1"
model_version = 2

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

data = ["I am so happy to be here"]
df = pd.DataFrame(data)


print(model.predict(df))
