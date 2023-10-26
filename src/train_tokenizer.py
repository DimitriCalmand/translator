from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from time import time
import pickle

from utils import *

df = pd.read_csv('../data/big_translate.csv', chunksize = 100000)

tokenizer_fr = Tokenizer(
        num_words=NUM_WORDS,
        oov_token = "<oov>",
        )
tokenizer_en = Tokenizer(
        num_words=NUM_WORDS,
        oov_token = "<oov>",
        )

i = 0
tps = time()
for data in df:
    fr, en = data['fr'].fillna(''), data['en'].fillna('')  # Remplacez les valeurs NaN par des chaînes vides
    fr, en = fr.tolist(), en.tolist()
    fr = [START_WORD + ' ' + sentence + ' ' + END_WORD for sentence in fr]
    en = [START_WORD + ' ' + sentence + ' ' + END_WORD for sentence in en]
    tokenizer_fr.fit_on_texts(fr)
    tokenizer_en.fit_on_texts(en)
    if i % 10 == 0:
       print(i*100000, "/", 12000000) 
    i += 1

print(tokenizer_en.texts_to_sequences(["<start> I how are you ! <end>"]))
# Enregistrement du tokenizer à l'aide de pickle
with open('saver/tokenizer_fr.pkl', 'wb') as f:
    pickle.dump(tokenizer_fr, f)
with open('saver/tokenizer_en.pkl', 'wb') as f:
    pickle.dump(tokenizer_en, f)

print("len config = ", len(tokenizer_en.word_index))
print("time exec = ", time() - tps)
