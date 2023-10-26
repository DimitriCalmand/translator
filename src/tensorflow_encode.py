import pandas as pd
import tensorflow as tf

# Chargement du fichier CSV avec pandas pour obtenir la taille du vocabulaire
def load1():
    df = pd.read_csv('../data/translate.csv')
    
    french = df["fr"][:100]
    english = df["en"][:100]
    
    # Création du tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words = 1000,
            filters='')
    tokenizer.fit_on_texts(french + english)
    
    # Chargement du fichier CSV dans TensorFlow
    dataset = tf.data.experimental.make_csv_dataset('../data/translate.csv', batch_size=9, num_epochs=1,
                                                   num_parallel_reads=5, shuffle_buffer_size=10000)
    return tokenizer, dataset    

import tensorflow as tf
import tensorflow_text as text
from encode import Tokenizer, load
# Créer le tokenizer
path = "/home/dimitri/documents/python/ia/nlp/"+ \
            "translator/data/translate.csv"
french, english = load(path)
tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words = 1000,
            filters='')
tokenizer.fit_on_texts(french + english)

# Définir la fonction de prétraitement avec le tokenization
def preprocess(row):
    fr_text = row['fr']
    en_text = row['en']

    fr_tokens = tokenizer.texts_to_sequences([fr_text.numpy().decode('utf-8')])[0]
    en_tokens = tokenizer.texts_to_sequences([en_text.numpy().decode('utf-8')])[0]

    return fr_tokens, en_tokens
# Charger le dataset CSV
dataset = tf.data.experimental.make_csv_dataset(
    '../data/translate.csv',  # Chemin vers votre fichier CSV
    batch_size=32,
    column_names=['fr', 'en'],  # Noms de vos colonnes
    label_name=None,  # Ajuster en conséquence si vous avez une colonne de libellé
    num_epochs=1,
    header=True
)

# Appliquer la fonction de prétraitement aux données chargées
dataset = dataset.map(preprocess)

# Parcourir le dataset tokenizé
for data in dataset:
    print(data)

