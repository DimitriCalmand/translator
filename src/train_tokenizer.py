from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from time import time
import pickle
import h5py as h5

from utils import *
def train_tokenizer():
    df = pd.read_csv('../data/translate.csv', chunksize = CHUNK_SIZE)
    
    tokenizer_fr = Tokenizer(
            num_words=NUM_WORDS,
            oov_token = "<oov>",
            )
    tokenizer_en = Tokenizer(
            num_words=NUM_WORDS,
            oov_token = "<oov>",
            )
    
    i = 0
    
    for data in df:
        fr, en = data['fr'].fillna(''), data['en'].fillna('')  # Remplacez les valeurs NaN par des chaînes vides
        fr, en = fr.tolist(), en.tolist()
        fr = [START_WORD + ' ' + sentence + ' ' + END_WORD for sentence in fr]
        en = [START_WORD + ' ' + sentence + ' ' + END_WORD for sentence in en]
        tokenizer_fr.fit_on_texts(fr)
        tokenizer_en.fit_on_texts(en)
        if i % 10 == 0:
            print(i*CHUNK_SIZE, "/", 12000000) 
        i += 1
    
    with open('saver/tokenizer_fr.pkl', 'wb') as f:
        pickle.dump(tokenizer_fr, f)
    with open('saver/tokenizer_en.pkl', 'wb') as f:
        pickle.dump(tokenizer_en, f)

def preprocess(old_path, new_path):
    with open(old_path, 'r') as fp:
        nb_line = sum(1 for line in fp if line.rstrip()) - 1
    print("number of line is :", nb_line)
    shape = (2, nb_line, MAX_LENGHT)
    chunk_shape = (2, CHUNK_SIZE, MAX_LENGHT)
    df = pd.read_csv(old_path, chunksize = CHUNK_SIZE)
    f = h5.File(
            new_path,
            'w',
            rdcc_nbytes = 1024 ** 2 * 4000,
            rdcc_nslots = 1e7
            )
    d = f.create_dataset(
            'data',
            shape,
            dtype = np.int32,
            chunks = chunk_shape
            )
    print("file create")
    i = 0
    for data in df:
        fr, en = data['fr'].fillna(''), data['en'].fillna('') 

        fr = [START_WORD + ' ' + sentence + ' ' + END_WORD for sentence in fr]
        en = [START_WORD + ' ' + sentence + ' ' + END_WORD for sentence in en]

        inputs = tokenizer_fr.texts_to_sequences(fr) 
        outputs = tokenizer_en.texts_to_sequences(en)

        inputs = pad_sequences(
                inputs, 
                maxlen=MAX_LENGHT, 
                padding='post', 
                truncating='post'
                )

        outputs = pad_sequences(
                outputs, 
                maxlen=MAX_LENGHT, 
                padding='post', 
                truncating='post'
                )
        res = np.array([inputs, outputs])
        d[:, i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE, :] =  res
        print(f"data {i}, is ok")
        i += 1
    f.close()
def main():
    train_tokenizer()
    preprocess("../data/translate.csv", "../data/preprocess.h5")
