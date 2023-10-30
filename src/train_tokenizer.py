from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
import numpy as np
import pandas as pd
from time import time
import pickle
import h5py as h5
from utils import *
from encode import *

def add_special_token(tokenizer):
    word_index = {
            '<oov>' : 1,
            START_WORD : 2,
            END_WORD : 3,
            NUMBER_WORD : 4,
            MAIL_WORD : 5,
            NAME_WORD : 6
            }
    for v, k in tokenizer.word_index.items():
        if  k == 1:
            continue
        word_index[v] = k + NB_SPECIAL_WORD
    index_word = {v: k for k, v in word_index.items()}
    tokenizer.word_index = word_index 
    tokenizer.index_word = index_word

def train_tokenizer():
    df = pd.read_csv('../data/translate.csv', chunksize = CHUNK_SIZE)
    tokenizer_fr = Tokenizer(
            num_words=NUM_WORDS,
            oov_token = "<oov>",
            filters='.'
            )
    tokenizer_en = Tokenizer(
            num_words=NUM_WORDS,
            oov_token = "<oov>",
            filters='.'
            )
    i = 0
    median = []
    for data in df:
        fr, en = data['fr'].fillna(''), data['en'].fillna('')  # Remplacez les valeurs NaN par des chaînes vides
        fr, en = fr.tolist(), en.tolist()
        for string in fr:
            string = string.split(' ')
            median.append(len(string))
        tokenizer_fr.fit_on_texts(fr)
        tokenizer_en.fit_on_texts(en)
        if i % 10 == 0:
            print(i*CHUNK_SIZE, "/", 12000000) 
        i += 1
    median.sort()
    print("median = ", median[len(median) // 2])
    print("moyenne = ", sum(median) / len(median))
    print("number word fr = ", len(tokenizer_fr.word_index))
    print("number word en = ", len(tokenizer_en.word_index))
    add_special_token(tokenizer_fr)
    add_special_token(tokenizer_en)

    with open('../data/saver/tokenizer_fr.pkl', 'wb') as f:
        pickle.dump(tokenizer_fr, f)
    with open('../data/saver/tokenizer_en.pkl', 'wb') as f:
        pickle.dump(tokenizer_en, f)

def preprocess(old_path, new_path):
    with open('../data/saver/tokenizer_fr.pkl', 'rb') as f:
        tokenizer_fr = pickle.load(f)
    with open('../data/saver/tokenizer_en.pkl', 'rb') as f:
        tokenizer_en = pickle.load(f)
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
        inputs = encode(fr.tolist(), 'fr')
        outputs = encode(en.tolist(), 'en')
        res = np.array([inputs, outputs])
        d[:, i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE, :] =  res
        print(f"data {i}, is ok")
        i += 1
    f.close()

def main():
    #train_tokenizer()
    #preprocess("../data/train.csv", "../data/train.h5")
    #preprocess("../data/test.csv", "../data/test.h5")
    dico = []
    res = encode(["il ert probablement"], dico)
    print(res)
    decoder = decode(res, tokenizer_fr, dico)
    print(decoder)
    
