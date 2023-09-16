import pandas as pd
import numpy as np
import tensorflow as tf

def load(path:str):
    df = pd.read_csv(path)
    return df
class tokenizer:
    def __init__(self,
            max_word = 10000,
            special_char = ",.?!':;"
            ):
        self.max_word = max_word 
        self.nb_word = nb_word
        self.special_char = special_char
        self.config = {}
        self.recurent_word = {}
    def train(self, input):
        for sentence in input:
            for char in special_char :
                sentence.replace(char, ' '+char)
            list_word = sentence.split(' ')
            for word in list_word:
                if word in self.recurent_word:
                    self.recurent_word[word] += 1
                else:
                    self.recurent_word[word] = 1
        sorted_keys = sorted(self.recurent_word,
                            key=lambda k:self.recurent_word[k])
        self.config["<oov>":1]
        for index,key in enumerate(reversed(sorted_keys)):
            self.config[index+2] = key
    def call(self, sentence):
        for char in special_char :
            sentence.replace(char, ' '+char)
        list_word = sentence.split(' ')
        res = []
        for word in list_word:
            res.append(self.config[list_word])
        return res

            
        


def encode_using_mine(sentence:str):
    special_char = 
    
   
def encode_using_hugging():
    """ man page
    https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Config """

    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenize_output = tokenizer(word)["input_ids"]
    return tokenize_output

    
    

if __name__ == "__main__":
    #df = load("/home/dimitri/documents/python/artificial_inteligence/nlp/"+
    #        "translator/data/translate.csv")
    #print(df.head())
    test()

