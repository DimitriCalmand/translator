import tensorflow as tf
import numpy as np

class embedding_layer(tf.keras.layers.Layer):
    def __init__(self,
            max_sequence_length,
            vocab_size,
            embed_dim,
            ):
        super(embedding_layer, self).__init__()
        self.token_embeddings = tf.keras.layers.Embedding(
                input_dim=vocab_size, 
                output_dim=embed_dim
                )
        self.position_embeddings = tf.keras.layers.Embedding(
                input_dim=max_sequence_length,
                output_dim=embed_dim
                )
    def call(self, inputs):
        #print(f"embedding{self.debug_count} ", tf.math.reduce_mean(inputs))
        lenght = tf.shape(inputs)[-1]
        positions = tf.range(lenght)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
class encoder_layer(tf.keras.layers.Layer):
    def __init__(self,
            num_heads,
            embed_dim,
            feed_forward_dim,
            dropout=0.0,
            ):
        """An encoder layer for a translater French -> English"""
        super(encoder_layer, self).__init__()
        self.multi_h_a = tf.keras.layers.MultiHeadAttention(
                num_heads,
                embed_dim,
                dropout=0.0)
        self.ffn = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(feed_forward_dim, activation="relu"),
                    tf.keras.layers.Dense(embed_dim)
                ]
            )
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
    def call (self, inputs, training=False):
        #print("encoder mean = ", tf.math.reduce_mean(inputs))
        multi_h_a = self.multi_h_a(inputs, inputs)
        norm1 = self.norm1(multi_h_a+inputs)
        dropout1 = self.dropout1(norm1, training = training)
        ffn = self.ffn(dropout1)
        norm2 = self.norm2(ffn + dropout1)
        dropout2 = self.dropout2(norm2, training = training)
        return dropout2

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self,
            nb_encoder,
            num_heads,
            embed_dim,
            feed_forward_dim,
            max_sequence_length,
            vocab_size,
            dropout=0.0
            ):
        super(TransformerEncoder, self).__init__()
        self.embedding_layer = embedding_layer(
                                            max_sequence_length,
                                            vocab_size,
                                            embed_dim
                                            )
        layers = []
        for i in range(nb_encoder):
            layers.append(
                    encoder_layer(
                        num_heads,
                        embed_dim,
                        feed_forward_dim,
                        dropout=dropout
                        )
                    )
        self.layers = layers

    def call(self, inputs, training = False):
        encoder = self.embedding_layer.call(inputs)
        #tf.print("mean input embeding encoder = ", tf.math.reduce_mean(tf.math.abs(encoder)))
        for layer in self.layers:
            encoder = layer(encoder, training=training)
        #tf.print("mean encoder = ", tf.math.reduce_mean(tf.math.abs(encoder)))
        #tf.print("encoder = ", encoder)
        return encoder
