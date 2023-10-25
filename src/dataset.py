import tensorflow as tf
import pandas as pd
from itertools import cycle
import os

class Dataset(tf.keras.utils.Sequence):
    def __init__(self,
            path_csv,
            tokenizer,
            batch_size,
            ):
        self.path_csv = path_csv
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        data = pd.read_csv(path_csv, chunksize=batch_size)
        self.data = cycle(data)
        len  = 0
        with open(path_csv, "rbU") as f:
            self.len = sum(1 for _ in f)
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        data = next(self.data)
        french = df["fr"]
        english = df["en"]
        return data


        
        

