my_dict = {"chat": 3, "chien": 1, "oiseau": 2, "poisson": 4}

# Tri du dictionnaire par les valeurs et obtention des clés triées
sorted_keys = sorted(my_dict, key=lambda k: my_dict[k])

# Création d'un nouveau dictionnaire avec les clés triées et des indices
indexed_dict = {key: index for index, key in enumerate(reversed(sorted_keys))}

print(indexed_dict)

