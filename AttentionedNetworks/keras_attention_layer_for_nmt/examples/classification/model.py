from tensorflow import keras
from AttentionedNetworks.keras_attention_layer_for_nmt.layers.attention import AttentionLayer
import tensorflow as tf
import numpy as np

def attention_lstm(hidden_size, max_len, max_features, v_size):
    # Input for variable-length sequences of integers
    encoder_inputs = keras.Input(shape=(max_len,), dtype="int32", name='inputs')
    # Embed each integer in a 128-dimensional vector
    x = keras.layers.Embedding(max_features, v_size, name='embedding_layer', input_length=max_len)(encoder_inputs)
    # Add 2 bidirectional LSTMs
    encoder_lstm = keras.layers.GRU(hidden_size, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_out, encoder_state = encoder_lstm(x)
    encoder_state = keras.layers.Reshape((1, hidden_size), input_shape=(hidden_size,))(encoder_state)

    attn_layer = AttentionLayer(name='attention_layer')
    atten_out, attn_stats = attn_layer([encoder_out, encoder_state])
    # Concat attention input and decoder GRU output
    decoder_concat_input = keras.layers.Concatenate(axis=-1, name='concat_layer')([attn_stats, atten_out])

    # Add a classifier
    outputs = keras.layers.Dense(1, activation="sigmoid")(decoder_concat_input)
    model = keras.Model(encoder_inputs, outputs)
    model.summary()
    return model


# the code I implemented is not correct
def attention_lstm_2(hidden_size, max_len, max_features, v_size):
    # Input for variable-length sequences of integers
    encoder_inputs = keras.Input(shape=(max_len,), dtype="int32", name='inputs')
    # Embed each integer in a 128-dimensional vector
    x = keras.layers.Embedding(max_features, v_size, name='embedding_layer', input_length=max_len)(encoder_inputs)
    # encoder LSTMs
    encoder_lstm = keras.layers.GRU(hidden_size, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_out, encoder_state = encoder_lstm(x)
    encoder_state = keras.layers.Reshape((-1, hidden_size))(encoder_state)

    decoder_inputs = keras.Input(shape=(1,), dtype="int32", name='decoder_inputs')
    y = keras.layers.Embedding(1, v_size, name='embedding_layer')(decoder_inputs)
    decoder = keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_out, decoder_state = decoder(decoder_inputs, initial_state=encoder_state)

    attn_layer = AttentionLayer(name='attention_layer')
    atten_out, attn_stats = attn_layer([encoder_out, encoder_state])
    # Concat attention input and decoder GRU output
    # decoder_concat_input = keras.layers.Concatenate(axis=-1, name='concat_layer')([attn_stats, atten_out])

    # Add a classifier
    # outputs = keras.layers.Dense(1, activation="sigmoid")(decoder_concat_input)
    outputs = keras.layers.Dense(1, activation="sigmoid")(atten_out)
    model = keras.Model(encoder_inputs, outputs)
    model.summary()
    return model

def lstm(hidden_size, max_len, max_features, v_size):
    # Input for variable-length sequences of integers
    encoder_inputs = keras.Input(shape=(None,), dtype="int32", name='inputs')
    # Embed each integer in a 128-dimensional vector
    x = keras.layers.Embedding(max_features, v_size, name='embedding_layer')(encoder_inputs)
    # Add 2 bidirectional LSTMs
    encoder_lstm = keras.layers.GRU(hidden_size, name='encoder_lstm')
    encoder_state = encoder_lstm(x)

    # Add a classifier
    outputs = keras.layers.Dense(1, activation="sigmoid")(encoder_state)
    model = keras.Model(encoder_inputs, outputs)
    model.summary()
    return model
