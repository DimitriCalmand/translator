import tensorflow as tf
import pandas as pd
from itertools import cycle
import numpy as np
import os
from encode import * 

def _preprocess(self, data, tokenizer):
    fr = tf.strings.regex_replace(data, "old_string", "new_string")
    en = tf.strings.regex_replace(data, "How", "new_string")
    return en 
def _preprocess_map(self,x, tokenizer):
    return _preprocess(x, tokenizer)

tokenizer = Tokenizer(get_file = True)
class CustomDataset(tf.data.Dataset):
    def _generator(input_file_path, chunksize):
        input_file_path = input_file_path.decode("utf-8")
        df = pd.read_csv(input_file_path, chunksize=chunksize);
        for row in df:
            inputs = np.array(tokenizer.call(row['fr'].tolist()))
            outputs = np.array(tokenizer.call(row['en'].tolist()))
            res = (
                {"encoder_inputs":inputs[:, 1:], "decoder_inputs":outputs[:, :-1]},
                {"outputs": outputs[:, 1:]}
                )

            yield res 

    def __new__(cls, input_file_path, batch_size=100):
        dataset = tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                {
                    "encoder_inputs": tf.TensorSpec(shape=(batch_size, 76), dtype=tf.int32),
                    "decoder_inputs": tf.TensorSpec(shape=(batch_size, 76), dtype=tf.int32)
                    }, 
                {
                    "outputs": tf.TensorSpec(shape=(batch_size, 76), dtype=tf.int32),  
                    }
            ),
            args=(input_file_path, batch_size)
        )
        dataset = dataset.prefetch(9).cache()
        return dataset
