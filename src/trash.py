import h5py
import numpy as np

# Création des données pour l'exemple
data_chunk = 1000000
data_total = 12000000
data = np.random.random((data_chunk, 200))

# Sauvegarde du premier morceau dans un fichier HDF5
with h5py.File('data.h5', 'w') as hf:
    dset = hf.create_dataset('data', data=data, maxshape=(data_total, 200), chunks=(data_chunk, 200))

# Ajouter des données par morceaux
with h5py.File('data.h5', 'a') as hf:
    for i in range(1, data_total // data_chunk):
        data = np.random.random((data_chunk, 200))
        hf["data"].resize(hf["data"].shape[0] + data.shape[0], axis=0)
        hf["data"][-data.shape[0]:] = data

# Vérification des formes
from time import time
tps = time()
with h5py.File('data.h5', 'r') as hf:
    data_loaded = hf["data"][:]
    data_loaded = np.array(data_loaded)

# Conversion en format Keras dataset
data_loaded = np.expand_dims(data_loaded, axis=0)
print(data_load.shape)
print(time() - tps)
# Vérification des formes
print(data_loaded)  # Assurez-vous que la forme est bien (3, 10)
def main():
    pass

