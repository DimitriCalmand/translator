import tensorflow as tf
import keras

from utils import *
def remove_zeros(tensor):
    tensor = tf.boolean_mask(tensor, tf.not_equal(tensor, 0))
    return tensor
def pretty_print(liste):
    liste = liste.split(" ")
    res = ""
    for string in liste:
        if (string == START_WORD or string == END_WORD):
            continue
        res += string + " "
    return res
def decode(tensor, tokenizer):
    tensor = [remove_zeros(tensor).numpy()]
    liste = tokenizer.sequences_to_texts(tensor)
    return pretty_print(liste[0])

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
            for i in range(4):
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
