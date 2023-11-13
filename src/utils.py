import pickle
from tensorflow.keras.utils import pad_sequences
from _config_test import *


MAX_LENGHT = 7 
BATCH_SIZE = 7 
NUM_WORDS = 10000 
START_WORD = '<start>'
END_WORD = '<end>'
NUMBER_WORD = '<number>'
MAIL_WORD = '<mail>'
NAME_WORD = '<name>'

NB_SPECIAL_WORD = 5
CHUNK_SIZE = 10000 
if (True):
    with open('../data/saver/tokenizer_fr.pkl', 'rb') as f:
        tokenizer_fr = pickle.load(f)
    with open('../data/saver/tokenizer_en.pkl', 'rb') as f:
        tokenizer_en = pickle.load(f)
    tokenizer_en.num_words = NUM_WORDS
    tokenizer_fr.num_words = NUM_WORDS
    START_TOKEN = tokenizer_en.word_index[START_WORD] 
    END_TOKEN = tokenizer_en.word_index[END_WORD] 

