import tensorflow as tf
import pandas as pd
import numpy as np
from utils import *
from time import time

class CustomDataset(tf.data.Dataset):
    def _generator(input_file_path, chunksize):
        input_file_path = input_file_path.decode("utf-8")
        df = pd.read_csv(input_file_path, chunksize=chunksize);
        for data in df:
            tps = time()            
            fr, en = data['fr'].fillna(''), data['en'].fillna('')  # Remplacez les valeurs NaN par des chaînes vides
            fr, en = fr.tolist(), en.tolist()

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
            print("\ntime = ", time() - tps, "\n")
            res = (
                {"encoder_inputs":inputs[:, 1:], "decoder_inputs":outputs[:, :-1]},
                {"outputs": outputs[:, 1:]}
                )

            yield res 
    def __new__(cls, input_file_path, batch_size=BATCH_SIZE):
        dataset = tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                {
                    "encoder_inputs": tf.TensorSpec(shape=(None, MAX_LENGHT - 1), dtype=tf.int32),
                    "decoder_inputs": tf.TensorSpec(shape=(None, MAX_LENGHT - 1), dtype=tf.int32)
                    }, 
                {
                    "outputs": tf.TensorSpec(shape=(None, MAX_LENGHT - 1), dtype=tf.int32),  
                    }
            ),
            args=(input_file_path, batch_size)
        )
        dataset = dataset.prefetch(9).cache()
        return dataset
