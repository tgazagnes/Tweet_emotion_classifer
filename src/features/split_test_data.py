import pandas as pd
import os

# preprocessing

PATH = r"C:\Users\thiba\Documents\Projets data\202402_NLP_emotions\data"

if __name__ == "__main__":

    fichier = "text.csv"
    try:
        df = pd.read_csv(os.path.join(PATH, fichier), index_col=0)
    except Exception as e:
        print("Unable to load the csv base", e)

    df_test = df.sample(frac=0.2, random_state=42)
    df_train = df.drop(df_test.index, axis=0)

    # Enregistrement jeu d'entraînement
    df_train.to_csv(os.path.join(PATH, "text_train.csv"))
    # Enregistrement jeu de test
    df_test.to_csv(os.path.join(PATH, "text_test.csv"))
    len_train = len(df_train)
    len_test = len(df_test)
    print(
        f"Jeux d'entraîement et de test enregistrés : \
            TRAIN {len_train} lignes, TEST {len_test} lignes"
    )
