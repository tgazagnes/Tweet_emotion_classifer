import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

# Import data
PATH = r"C:\Users\thiba\Documents\Projets data\202402_NLP_emotions\data"
fichier = "text_test.csv"

# Import labels
target_labels = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

# Page setting : wide layout
st.set_page_config(page_title="Analyse de sentiment réseaux sociaux")

# Configuration du menu gauche
st.sidebar.write("**Analyse de sentiment à partir de tweets (en anglais)**")
st.sidebar.write("")
st.sidebar.write(
    """
    Dataset : [Nidula Elgiriyewithana]\
        (https://www.kaggle.com/datasets/nelgiriyewithana/emotions)

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
    """# 🔎 Analyse de sentiment sur les réseaux sociaux (tweets)
"""
)


# Interface : choisir de charger un fichier ou de sélectionner un échantillon de test
cell = st.container()
with cell:
    choix_mode = st.radio(
        "Choisir un mode de prédiction :",
        [
            "A partir d'un fichier csv",
            "A partir d'un échantillon de test",
            "Ecrire un tweet (en anglais)",
        ],
        index=None,
    )

cell_2 = st.container()
with cell_2:

    if choix_mode is None:
        st.caption("(Veuillez sélectionner une option ci-dessus.)")

    elif choix_mode == "A partir d'un fichier au format csv":
        file_input = st.file_uploader("Choisir un fichier .csv", type="csv")
        if file_input is not None:
            # Can be used wherever a "file-like" object is accepted:
            input_df = pd.read_csv(file_input)
            st.write(input_df)

    # Code pour la prédiction à partir d'un échantillon
    elif choix_mode == "A partir d'un échantillon de test":
        liste_echantillons = os.listdir(os.path.join(PATH, "test_samples"))
        test_sample = st.selectbox(
            "Choisir un échantillon dans la liste", liste_echantillons
        )

        # check if filepath
        filepath = os.path.join(PATH, "test_samples", test_sample)
        assert os.path.isfile(filepath)
        with open(filepath, "r") as f:
            pass
        df = pd.read_csv(filepath, index_col=0)

        # Random à changer pour la pred
        df["Prédiction"] = np.random.randint(0, 6, len(df))

        df_count_true = df["label"].value_counts().sort_index()
        df_count_pred = df["Prédiction"].value_counts()
        df_count = pd.concat([df_count_true, df_count_pred], axis=1)
        df_count = df_count.rename({"label": "Réel"}, axis=1)
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
        score_prediction = "XXX %"
        cell2.metric("Justesse de la prédiction", f"{score_prediction}")

        fig2 = px.bar(
            df_count,
            x="Sentiment",
            y=["Réel", "Prédiction"],
            title="Répartition des tweets par sentiment",
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

        # debug
        st.dataframe(df_count)
        # debug
        st.dataframe(df)

    # Code pour la prédiction à partir de la saisie manuelle d'un tweet
    else:
        text_input = st.text_area(
            "Tweet à analyser", "Write your tweet in english here"
        )

        st.write(f"You wrote {len(text_input)} characters.")


# @st.cache_data
# def load_data():
#    df = pd.read_parquet("data/base_consolidee.parquet")
# Copie des données pour transfo
#    df_total = df.copy()
#    return df_total

# df_total = load_data()
