# NLP - Analyse de sentiments sur les réseaux sociaux (MLflow, Streamlit)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) 
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) 
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white) 

![Pre-commit passed](https://results.pre-commit.ci/badge/github/pre-commit/pre-commit/main.svg)

## Motivation

Ce projet vise à entraîner un modèle de classification de texte sur la base de tweets afin d'analyser le sentiment principal qui s'en dégage.
Pour entraîner et choisir un modèle pertinent, j'ai utilisé le framework [mlflow](https://github.com/mlflow) qui permet de tracker et comparer les expérimentations successives pour retenir le meilleur modèle. L'interface pour tester les prédictions du modèle a été développée sur [streamlit](https://github.com/streamlit) qui est mon favori du moment.

## Aperçu
![Capture streamlit](https://github.com/tgazagnes/Tweet_emotion_classifer/blob/main/reports/Capture1.PNG?raw=true)


## References

## Organisation du répertoire

    ├── LICENSE
    ├── README.md          
    ├── data               <- conservée en local uniquement
    ├── mlruns             <- Expérimentations MLFLOW et modèles retenus pour déploiement
    ├── notebooks          <- Notebooks d'exploration et de tests
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── features       <- Scripts de séparation des données d'entraînement et tests    │   │
    │   ├── models         <- Scripts d'entraînement des modèles trackés avec MLFlow
    │   ├── streamlit      <- Scripts de création de l'app sur Streamlit    


## Prochaines étapes (à venir)
- Optimisation des hyperparamètres (Optuna)
- Containerisation Docker

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
