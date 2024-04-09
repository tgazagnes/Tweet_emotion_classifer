import pandas as pd
import os

PATH = r"C:\Users\thiba\Documents\Projets data\202402_NLP_emotions\data"

if __name__ == "__main__":

    fichier = "text_test.csv"
    try:
        df = pd.read_csv(os.path.join(PATH, fichier), index_col=0)
    except Exception as e:
        print("Unable to load the csv base", e)

    for i in range(3):
        title = "Echantillon_de_test_" + str(i) + ".csv"
        df_sample = df.sample(frac=0.2, random_state=(12 * i))

        # Enregistrement
        df_sample.to_csv(os.path.join(PATH, "test_samples", title))
        len_test = len(df_sample)
        print(f"Echantillons de tests extrait : {len_test} lignes")
