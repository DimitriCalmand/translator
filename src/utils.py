import pickle
from tensorflow.keras.utils import pad_sequences



MAX_LENGHT = 25 
BATCH_SIZE = 64
NUM_WORDS = 4000 
START_WORD = '<start>'
END_WORD = '<end>'
NUMBER_WORD = '<number>'
MAIL_WORD = '<mail>'
NAME_WORD = '<name>'

NB_SPECIAL_WORD = 5
CHUNK_SIZE = 1000 
if (True):
    with open('saver/tokenizer_fr.pkl', 'rb') as f:
        tokenizer_fr = pickle.load(f)
    with open('saver/tokenizer_en.pkl', 'rb') as f:
        tokenizer_en = pickle.load(f)
    START_TOKEN = tokenizer_en.word_index[START_WORD] 
    END_TOKEN = tokenizer_en.word_index[END_WORD] 

