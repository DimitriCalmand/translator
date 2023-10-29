import tensorflow as tf
import keras
from encode import *
from utils import *

class DisplayOutputs(keras.callbacks.Callback):
    def __init__(self, batch, verbose = 10, model = None):
        """Displays a batch of outputs after every epoch"""
        self.verbose = verbose
        self.batch = batch
        if model is not None:
            self.model = model

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self. verbose == 0:
            batch = next(self.batch)
            print()
            for i in range(1):
                source = batch[0]["encoder_inputs"].numpy()[i:i+1]
                target = batch[0]["decoder_inputs"].numpy()[i:i+1]
                predicted = self.model.generate(source)

                str_predicted = decode(predicted, tokenizer_en)
                str_target = decode(target, tokenizer_en)
                str_source = decode(source, tokenizer_fr)

                print('-'*80)
                print('source : ', str_source)
                print("target : ", str_target)
                print('Prediction: ', str_predicted)
            print('-'*80)
