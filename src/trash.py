import tensorflow as tf

# Créez un ensemble de données d'exemples numériques simples
dataset = tf.data.Dataset.range(1, 100)  # Un ensemble de données de 1 à 10

# Définissez la taille du tampon de mélange
BUFFER_SIZE = 20 

# Mélangez l'ensemble de données en utilisant BUFFER_SIZE
shuffled_dataset = dataset.shuffle(BUFFER_SIZE)

# Parcourez et affichez les éléments mélangés
for element in shuffled_dataset:
    print(element.numpy())

