import pickle
from tensorflow.keras.utils import pad_sequences



MAX_LENGHT = 200
BATCH_SIZE = 64
NUM_WORDS = 10000 
START_WORD = 'starttoken'
END_WORD = 'endtoken'
CHUNK_SIZE = 10
if (True):
    with open('saver/tokenizer_fr.pkl', 'rb') as f:
        tokenizer_fr = pickle.load(f)
    with open('saver/tokenizer_en.pkl', 'rb') as f:
        tokenizer_en = pickle.load(f)
    
    START_TOKEN = tokenizer_en.word_index[START_WORD] 
    END_TOKEN = tokenizer_en.word_index[END_WORD] 

