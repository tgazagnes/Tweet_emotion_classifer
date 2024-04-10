import streamlit as st
import pandas as pd
import plotly.express as px
import os
import mlflow
from sklearn.metrics import f1_score

# Import data
PATH = r"C:\Users\thiba\Documents\Projets data\202402_NLP_emotions\data"
fichier = "text_test.csv"


# Import labels
target_labels = {
    0: "Triste",
    1: "Joyeux.se",
    2: "Amoureux.se",
    3: "En colère",
    4: "Apeuré.ée",
    5: "Surpris.e",
}

# import model from registry
model_name = "Sentiment_classifier_GBC"
model_version = 4
model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")


# Page setting : wide layout
st.set_page_config(page_title="Analyse de sentiment réseaux sociaux")

# Configuration du menu gauche
st.sidebar.write("**Analyse de sentiment à partir de tweets (en anglais)**")
st.sidebar.write("")
st.sidebar.write(
    """
    Dataset : [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/emotions)

    Stack : 

    ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 
    ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) 
    ![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
    ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
    ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)

    \n
    A propos : \n
    * Thibaut Gazagnes : [site](https://tgazagnes.github.io/) / \
        [linkedin](https://www.linkedin.com/in/thibautgazagnes)
    * [Github du projet](https://github.com/tgazagnes/Tweet_emotion_classifer).
    """
)


# Titre
st.markdown(
    """## 🔎 Analyse de sentiments sur les réseaux sociaux (tweets)
"""
)


# Interface : choisir de charger un fichier ou de sélectionner un échantillon de test
cell = st.container()
with cell:
    choix_mode = st.radio(
        "Choisir un mode de prédiction :",
        [
            "A partir d'un échantillon de test",
            "Ecrire un tweet (en anglais)",
        ],
        index=None,
    )

cell_2 = st.container()
with cell_2:
    if choix_mode is None:
        st.caption("(Veuillez sélectionner une option ci-dessus.)")

    # Code pour la prédiction à partir d'un échantillon
    elif choix_mode == "A partir d'un échantillon de test":
        liste_echantillons = os.listdir(os.path.join(PATH, "test_samples"))
        test_sample = st.selectbox(
            "Choisir un échantillon dans la liste", liste_echantillons, index=None
        )
        st.caption("NB : ces données n'ont pas servi pour l'entraînement du modèle.")

        if test_sample is None:
            st.empty()
        else:
            # check if filepath
            filepath = os.path.join(PATH, "test_samples", test_sample)
            assert os.path.isfile(filepath)
            with open(filepath, "r") as f:
                pass
            df = pd.read_csv(filepath, index_col=0)
            df["label"] = df["label"].astype("int")

            # Test on sample
            pred = model.predict(df["text"])
            df["pred"] = pred

            # Score de prédiction F1
            pred_score = f1_score(df["label"], pred, average="weighted")

            # Prepare data for graphs
            df_count_true = df["label"].value_counts().sort_index()
            df_count_true = df_count_true.rename("nb_reel")

            df_count_pred = df["pred"].value_counts()
            df_count_pred = df_count_pred.rename("nb_pred")

            df_count = pd.concat([df_count_true, df_count_pred], axis=1)
            #            st.dataframe(df_count)

            df_count["Sentiment"] = df_count.index
            df_count["Sentiment"] = df_count["Sentiment"].replace(target_labels)

            # Ligne 1 : 2 cellules avec les indicateurs clés en haut de page
            l1_col1, l1_col2 = st.columns(2)

            # Pour avoir la bordure, il faut nester un st.container dans chaque colonne

            # 1ère métrique
            cell1 = l1_col1.container(border=True)
            # Trick pour séparer les milliers
            nb_tweets = len(df)
            nb_tweets = f"{nb_tweets:,.0f}".replace(",", " ")
            cell1.metric("Nombre de tweets analysés", f"{nb_tweets}")

            # 2ème métrique
            cell2 = l1_col2.container(border=True)
            cell2.metric("Score de prédiction (F1-score)", f"{pred_score:.1%}")

            # graphique radar
            fig = px.line_polar(
                df_count,
                r="nb_pred",
                theta="Sentiment",
                line_close=True,
                title="Prédictions par sentiment (en nombre de tweets)",
            )
            fig.update_traces(fill="toself")
            st.plotly_chart(fig, use_container_width=True)

            # graphique à barres réel vs prédiction
            fig2 = px.bar(
                df_count,
                x="Sentiment",
                y=["nb_pred", "nb_reel"],
                title="Nombre de tweets par sentiment : comparaison réel/prédiction",
                text_auto=True,
                barmode="group",
            )
            fig2.update_traces(textposition="inside")
            fig2.update_layout(
                autosize=True,
                uniformtext_minsize=10,
                uniformtext_mode="hide",
                xaxis_title="Sentiment",
                yaxis_title="Nombre de tweets",
            )

            # Affichage du graphique
            st.plotly_chart(fig2, use_container_width=True)

            with st.expander("Voir le détail : "):
                # debug
                # st.dataframe(df_count)
                # debug
                st.dataframe(df)

    # Code pour la prédiction à partir de la saisie manuelle d'un tweet
    else:
        text_input = st.text_area(
            "Ecrire un tweet et cliquer sur 'Prédiction'",
            value="Example : Streamlit is so cool",
            max_chars=140,
        )

        st.write(f"You wrote {len(text_input)} characters.")

        if st.button("Prédiction"):

            cell_3 = st.container(border=True)

            with cell_3:

                # import dans un df
                df3 = pd.DataFrame({"text": text_input}, index=[0])
                # Prédiction à partir du modèle (à partir d'une série et pas d'un df)
                pred = model.predict(df3["text"])

                target_emoji = {
                    0: "Triste :cry:",
                    1: "Joyeux.se :joy:",
                    2: "Amoureux.se :heart_eyes:",
                    3: "En colère :persevere:",
                    4: "Apeuré.ée :fearful:",
                    5: "Surpris.e :astonished:",
                }

                target_label = target_emoji[int(pred)]

                st.title(f"{target_label}")

        else:
            st.caption("Ecrire votre tweet et cliquer sur Prédire")

# @st.cache_data
# def load_data():
#    df = pd.read_parquet("data/base_consolidee.parquet")
# Copie des données pour transfo
#    df_total = df.copy()
#    return df_total

# df_total = load_data()
