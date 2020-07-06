from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from AttentionedNetworks.keras_self_attention.attention_layers.seq_self_attention import SeqSelfAttention
from AttentionedNetworks.keras_self_attention.attention_layers.scaled_dot_attention import ScaledDotProductAttention
from AttentionedNetworks.keras_self_attention.attention_layers.seq_weighted_attention import SeqWeightedAttention


def seq_self_attention(hidden_size, max_len, max_features, v_size):
    inp = Input(shape=(None,), dtype="int32", name='inputs')
    x = Embedding(max_features, v_size, mask_zero=False)(inp)
    encoder_lstm = LSTM(hidden_size, return_sequences=True, name='encoder_lstm')
    x = encoder_lstm(x)
    x = SeqSelfAttention(attention_activation='sigmoid')(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inp, outputs)
    model.summary()
    return model


def scaled_dot_self_attention(hidden_size, max_len, max_features, v_size):
    inp = Input(shape=(None,), dtype="int32", name='inputs')
    x = Embedding(max_features, v_size, mask_zero=True)(inp)
    encoder_lstm = LSTM(hidden_size, return_sequences=True, name='encoder_lstm')
    x = encoder_lstm(x)
    x = ScaledDotProductAttention(name='Attention')(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inp, outputs)
    model.summary()
    return model


def seq_weighted_self_attention(hidden_size, max_len, max_features, v_size):
    inp = Input(shape=(None,), dtype="int32", name='inputs')
    x = Embedding(max_features, v_size, mask_zero=False)(inp)
    encoder_lstm = LSTM(hidden_size, return_sequences=True, name='encoder_lstm')
    x = encoder_lstm(x)
    x = SeqWeightedAttention()(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inp, outputs)
    model.summary()
    return model
