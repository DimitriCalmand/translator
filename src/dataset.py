import tensorflow as tf
import pandas as pd
import numpy as np
from utils import *
from time import time
import h5py as h5

class CustomDataset(tf.data.Dataset):
    def _generator(input_file_path, batch_size):
        input_file_path = input_file_path.decode("utf-8")
        f = h5.File(
                input_file_path,
                'r',
                rdcc_nbytes = 1024 ** 2 * 4000, #4Go
                rdcc_nslots = 1e7
                )
        d = f["data"]
        size = d.shape[1]
        for j in range(size // batch_size):
            data = np.array(d[:, j * batch_size: (j + 1) * batch_size, :])
            inputs = data[0, :, :]
            outputs = data[1, :, :] 
            res = (
                {"encoder_inputs":inputs[:, 1:], "decoder_inputs":outputs[:, :-1]},
                {"outputs": outputs[:, 1:]}
                )
            yield res 
        f.close()
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
